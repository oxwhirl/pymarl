local torch = require 'torch'
local class = require 'class'
local json = require 'json'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'
local csvigo = require 'csvigo'
require 'module.OneHot'

local tc = require 'torchcraft'
tc.DEBUG = 0
tc.mode.micro_battles = true
tc.mode.replay = false

local StarCraftMicro = class('StarCraftMicro')

-- connect to TorchCraft
function StarCraftMicro:__init(opt)
    self.opt = opt

    if self.opt.cuda == 1 then
        require 'cutorch'
    end

    self.first_match = false
    self.opt.bait = 0
    self.units_bait = {}
    self.current_dump = {}

    self.opt.game_grid_size_type = "standard"

    self.opt.nagents = self.opt.game_nagents
    -- Change this when we want to deal with different enemy numbers
    self.opt.nenemies = self.opt.game_nenemies

    self.hostname = self.opt.game_hostname
    self.port = self.opt.game_port

    self.reward_functions = {
        battle = function () return self:rewardBattle() end,
        individual = function () return self:rewardIndividual() end,
        individual_shifted = function () return self:rewardIndividual() end
    }

    self.state_functions = {
        grid = function () return self:gridState() end,
        action_range = function () return self:actionRangeState() end,
        feat = function (args) return self:featState(args) end
    }

    self.action_functions = {
        id_based = function (index, uid) return self:indexAttack(index, uid) end,
        relative = function (index, uid) return self:relativeAttack(index, uid) end
    }

    self.reward_function = self.reward_functions[self.opt.game_reward_function]
    self.state_function = self.state_functions[self.opt.game_state_function]
    self.action_function = self.action_functions[self.opt.game_action_function]
    assert(self.reward_function)
    assert(self.state_function)
    assert(self.action_function)

    self.opt.n_relative_attack_sections = 5

    -- can't have them together!
    assert(not (self.opt.game_state_health_own == 1 and self.opt.game_state_health_own_all == 1))

    self.grid_elem_size_height = self.opt.game_state_grid_height / self.opt.game_state_grid_coarseness_height
    self.grid_elem_size_width = self.opt.game_state_grid_width / self.opt.game_state_grid_coarseness_width
    -- check we have square aspect for grid. maybe not essential
    assert(self.grid_elem_size_width == self.grid_elem_size_height)
    self.opt.game_state_grid_expected_unit_size = self.grid_elem_size_height
    -- This is maximum size to distinguish agents reasonably well
    -- assert(self.grid_elem_size_height == self.opt.game_state_grid_expected_unit_size
    --            and self.grid_elem_size_width == self.opt.game_state_grid_expected_unit_size)

    self.tc = tc
    self.skip_frames = self.opt.game_skip_frames

    self.expected_enemy_units = self.opt.nenemies
    self.expected_myself_units = self.opt.nagents
    self.last_move_action_id = 5
    self.dead_action_ids = 1 + 4 + self.opt.nenemies + 1
    self.timeout_reward = -100

    self.auto_start_battle = true

    -- Totally Not Arbitrary(TM)
    self.state_boundaries = {
        north = 130,
        south = 150,
        west = 60,
        east = 110
    }

    -- set game info
    self.battles_won = 0
    self.battles_game = 0
    self.timeouts = 0

    -- actual setup
    self:setup()

    self.one_hot_type = nn.OneHot(2)

--  those are just for debugging - tracking rewards stats
    self.epc = 0
    self.total_nspikes = 0
    self.total_nshots = 0
    self.total_legataks = 0
    self.total_plus = 0
    self.total_minus = 0
    self.total_plus_leg = 0
    self.total_minus_leg = 0
    self.total_same = 0
    self.total_after = 0

    -- set battle info
    self:reset()


    -- Step Z E R O
    -- HACK this shouldn't be required, however doing it anyway as otherwise
    -- comm blows
    self:step(torch.zeros(1, self.opt.nagents) + self.last_move_action_id)
end


-- if game is on, leaves the game
-- then (or otherwise) starts another one
function StarCraftMicro:reset()
    -- Zero past information

    self.battle_steps = 1
    self.units_myself_map = {}
    self.units_enemy_map = {}

    self.death_tracker_myself = torch.zeros(self.opt.nagents)
    self.death_tracker_enemy = torch.zeros(self.opt.nenemies)
    self.previous_state_myself = nil
    self.previous_state_enemy = nil

    self.enemies_shot_by_agents = {}
    for i = 1, self.opt.nenemies do
        self.enemies_shot_by_agents[i] = {}
    end

    --    contains an estimate of the current angle for every agent
    self.theta = torch.zeros(self.opt.nagents)
    --    stores the last action taken
    self.last_a = nil
    --    saving the previous state once again for use in calculate_theta
    self.previous_state_enemy_for_state = nil

    self.last_effective_action_range = {}

    self.bait_random_action = torch.random(2, 3)
    self.save_run = (self.opt.game_save_runs == 1 and
                     self.battles_game % self.opt.game_save_runs_freq == 0)
    self:step(torch.zeros(1, self.opt.nagents) + self.last_move_action_id)

    self:debug_update_rewards_stats()

end

function StarCraftMicro:debug_update_rewards_stats()

--    if self.n_spikes then
--        self.epc = self.epc + 1
--
--        self.total_nspikes = self.total_nspikes + self.n_spikes
--        self.total_nshots = self.total_nshots + self.n_shots
--
--        self.total_legataks = self.total_legataks + self.n_legataks
--
--        self.total_same = self.total_same + self.n_at_same
--        self.total_after = self.total_after + self.n_at_after
--
--        local disc = self.n_spikes - self.n_shots
--        if disc > 0 then
--            self.total_plus = self.total_plus + 1
--        elseif disc < 0 then
--            self.total_minus = self.total_minus + 1
--        end
--
--        local disc_leg = self.n_spikes - self.n_legataks
--        if disc_leg > 0 then
--            self.total_plus_leg = self.total_plus_leg + 1
--        elseif disc_leg < 0 then
--            self.total_minus_leg = self.total_minus_leg + 1
--        end
--
--    end

    --    log.errorf("n_spikes %s, n_legataks %s, n_shots %s, n_at_same %s, n_at_after %s",self.n_spikes, self.n_legataks,self.n_shots, self.n_at_same, self.n_at_after)


    local total_same_after = self.total_same + self.total_after

    --    log.errorf("total_nspikes %s, total_nshots %s, total_plus %s, total_minus %s, perc_same %.2f, perc_after %.2f, total_leg_plus %.2f, total_leg_minus %.2f",
    --        self.total_nspikes,self.total_nshots, self.total_plus/self.epc, self.total_minus/self.epc,  self.total_same/total_same_after,
    --            self.total_after/total_same_after, self.total_plus_leg/self.epc, self.total_minus_leg/self.epc  )

    self.n_spikes = 0
    self.n_shots = 0
    self.n_legataks = 0
    self.n_at_same = 0
    self.n_at_after = 0

end


function StarCraftMicro:setup()
    -- No need to restart game, as we are in micro. Following is therefore
    -- potentially useless, but we need to send a message anyway.

    if self.game_is_running then
        local actions = {
            tc.command(tc.restart)
        }
        -- send restart command
        tc:send({table.concat(actions, ':')})

        while not tc.state.game_ended do
            local update = tc:receive()
            log.warn("waiting for end!")
        end
        tc:close()
        sys.sleep(0.5)
        collectgarbage()
        collectgarbage()
    end

    tc:init(self.hostname, self.port)
    local update = tc:connect(port)

    -- first message to BWAPI's side is setting up variables
    local setup = {
        tc.command(tc.set_speed, self.opt.game_speed), tc.command(tc.set_gui, 1),
        tc.command(tc.set_cmd_optim, 1),
        tc.command(tc.set_combine_frames, self.skip_frames)
    }
    tc:send({table.concat(setup, ':')})
    local update = tc:receive()

    -- Note: it should never go directly to battle_just_ended. If it blows up
    -- here it's because the game is in a weird state, and we do need to handle
    -- that case...
    while tc.state.waiting_for_restart do
        tc:send({table.concat(actions, ':')})
        update = tc:receive()
    end

    if self.opt.game_use_ext_config == 1 then
        self:setupFromConfig()
    end

    self.game_is_running = true

    return self
end

function StarCraftMicro:setupFromConfig()
    local map_name = string.match(tc.state.map_name, '^.+/(.+)$')
    if not map_name then
        map_name = tc.state.map_name
    end

    assert(map_name, 'Couldn\'t get a map name, instead got '
               .. map_name)

    local map_file = self.opt.game_ext_config_dir ..
        '/' .. map_name
    map_file = string.gsub(map_file, '//', '/')
    map_file = string.gsub(map_file, '.scm', '.json')

    self.opt.map = string.gsub(string.gsub(map_file, './config/', ''), '.json', '')
    self.dz_map = self.opt.map == 'dragoons_zealots' or self.opt.map == 'demo_map'

    self.unit_type_dict = {}
    if self.opt.map == '2w_4m' then
        self.unit_type_dict = {[0] = 1, [8] = 2}
    elseif self.opt.map == '2m_3zerg' then
        self.unit_type_dict = {[0] = 1, [37] = 2}
    end

    log.info("Loaded config from map file: " .. map_file)
    local config = json.load(map_file)
    assert(config, 'Config was not loaded correctly!')

    assert(config.name == string.gsub(tc.state.map_name, '.scm', '')
               and config.nagents and config.nenemies)

    self.opt.nagents = 0
    self.opt.nenemies = 0

    -- Ignoring t for now
    for t, val in pairs(config.nagents) do
        self.opt.nagents = self.opt.nagents + val
    end

    for t, val in pairs(config.nenemies) do
        self.opt.nenemies = self.opt.nenemies + val
    end

    if config.custom_grid then
        self.opt.game_grid_size_type = config.custom_grid.size
        self.opt.game_state_grid_height = config.custom_grid.height
        self.opt.game_state_grid_width = config.custom_grid.width
        self.opt.game_state_grid_coarseness_height = config.custom_grid.coarseness_height
        self.opt.game_state_grid_coarseness_width = config.custom_grid.coarseness_width
        -- TODO print new config
        -- TODO assert not nil!
    end

    if config.bait then
        self.opt.bait = config.bait
        self.state_boundaries = {
           north = 120,
           south = 160,
           west = 60,
           east = 110
        }


    end

    log.info("From config: [nagents: " .. self.opt.nagents
                 .. "], [nenemies: " .. self.opt.nenemies .. "]")
    self.expected_enemy_units = self.opt.nenemies
    self.expected_myself_units = self.opt.nagents
    self.opt.game_nenemies = self.opt.nenemies
    self.opt.game_nagents = self.opt.nagents
    self.dead_action_ids = 1 + 4 + self.opt.nenemies + 1
    self.opt.game_action_space = self.dead_action_ids
end

function StarCraftMicro:nObsFeature()
    if self.opt.game_state_function == 'feat' then
        local dummy_act, dummy_crit = self:featState()
        local actor_state_size = dummy_act[1][1]:size(2)
        local critic_state_size = dummy_crit[1][1]:size(2)
        return actor_state_size, critic_state_size
    elseif self.opt.game_state_function == 'action_range' then
        return self.opt.game_action_space, self.opt.game_action_space
    end

    if self.precomputed_nobsfeature then
        return self.precomputed_nobsfeature
    end

    local nobs_t = {}
    if self.opt.game_state_function == 'grid' or self.opt.game_state_function == 'action_range' then
        table.insert(nobs_t, {self.opt.game_state_grid_coarseness_height,
                              self.opt.game_state_grid_coarseness_width}) -- includes allies
        if self.opt.game_state_single_grid == 0 then
            table.insert(nobs_t, {self.opt.game_state_grid_coarseness_height,
                                  self.opt.game_state_grid_coarseness_width}) -- includes enemies
        end
    end

    if self.opt.model_dueling == 1 or self.opt.model_remix_qs == 1 then
        table.insert(nobs_t, {self.opt.game_state_full_state_grid_coarseness_height,
                              self.opt.game_state_full_state_grid_coarseness_width})
    end

    local health = 0
    if self.opt.game_state_health_own == 1 then
        health = health + 1
    elseif self.opt.game_state_health_own_all == 1 then
        health = health + self.opt.nagents
    end

    if self.opt.game_state_health_enemy == 1 then
        health = health + self.opt.nenemies
    end

    table.insert(nobs_t, {health})

    local nobs_sum

    -- Returning table here because don't have specs for when
    -- game state is not concatenated, yet.
    self.precomputed_nobsfeature = nobs_t
    return nobs_t
end

function StarCraftMicro:getActionRange(step, agent_id, overwrite_disable_action_range_opt, fov_scale)
    -- {agent_id: [action_id_k, ..., action_id_l]}
    -- If no enemies are in view, attacking is not available anymore If dead,
    -- only a "dead" action is available

--  this is passed for individual rewards
    fov_scale = fov_scale or self.opt.game_state_fov_scale

    -- TODO move move actions limit here

    -- local available_actions = tablex.range(1, self.last_move_action_id)
    local available_actions = {1}

    local uid
    if agent_id == -1 then
        uid = self.units_bait[1]
    else
        uid = self:mapIdtoTCId(agent_id, true)
    end

    local ut = tc.state.units_myself[uid]

    if not ut then
        available_actions = {self.dead_action_ids}
    elseif self.opt.game_disable_action_range==1 and not overwrite_disable_action_range_opt then
        for aid = 2, 1 + 4 + self.opt.nenemies do
            table.insert(available_actions, aid)
        end
    else
        if ut.position[2] > self.state_boundaries.north then
            table.insert(available_actions, 2)
        end
        if ut.position[2] < self.state_boundaries.south then
            table.insert(available_actions, 3)
        end
        if ut.position[1] > self.state_boundaries.west then
            table.insert(available_actions, 4)
        end
        if ut.position[1] < self.state_boundaries.east then
            table.insert(available_actions, 5)
        end
        if self.opt.game_action_function == "relative" then
            for i=1, self.opt.n_relative_attack_sections do
                local targets = self:getRelativeTargets(uid, i)
                if #targets > 0 then
                    table.insert(available_actions, self.last_move_action_id + i)
                end
            end
        elseif self.opt.game_action_function == "id_based" then
            for euid, eut in pairs(tc.state.units_enemy) do
                local distance = math.sqrt((ut.position[1] - eut.position[1])^2 +
                        (ut.position[2] - eut.position[2])^2)
                local enemy_in_range

                if self.dz_map then
--                  hardcode the range to be the same for dragoons and zealots, equal to dragoon real range
                    local gwrange = 16
                    enemy_in_range = distance <= gwrange*fov_scale
                else
                    local gwrange = math.max(16, ut.gwrange)
                    enemy_in_range = distance <= gwrange*fov_scale
                end

                if enemy_in_range then
                    local emapid = self:TCIdtoMapId(euid, false)
                    table.insert(available_actions, self.last_move_action_id + emapid)
                end
            end
        end
    end

    table.sort(available_actions)

    -- TODO: Make this work properly
    local action_range = {{{1},available_actions}}

    return action_range
end


function StarCraftMicro:getCommLimited(step, i)
    return
end


function StarCraftMicro:waitForBattleStart()

    while tc.state.battle_just_ended do
        tc:send({table.concat({}, ':')})
        local update = tc:receive()
    end

    for i=1, 100 do
        self.units_myself_map = self:mapUnits(tc.state.units_myself, true)
        self.units_enemy_map = self:mapUnits(tc.state.units_enemy)
        if (#self.units_myself_map == self.expected_myself_units
            and #self.units_enemy_map == self.expected_enemy_units
            and #self.units_bait == self.opt.bait) then
            return true
        else
            tc:send({table.concat({}, ':')})
            local update = tc:receive()
        end
    end
    return false
end


function StarCraftMicro:step(a)
    -- TODO check last index here...
    if ((self.auto_start_battle and tc.state.battle_just_ended)
            or #tablex.keys(self.units_myself_map) < 1
            or #tablex.keys(self.units_enemy_map) < 1)
    then
        -- If we want to automagically reset() we should do it here
        local result = self:waitForBattleStart()
        while not self:waitForBattleStart() do
            while not self:suicideStep() do end
        end
    end

    if (self.opt.game_force_battle_end == 1 and
            self.battle_steps >= self.opt.game_max_battle_steps) then
        while not self:suicideStep() do end

        local reward = (torch.zeros(1, self.opt.nagents):type(self.opt.dtype)
                            + self.timeout_reward)
        local terminal = torch.zeros(1) + 1
        -- battle assumed to have been lost
        self.battles_game = self.battles_game + 1
        self.timeouts = self.timeouts + 1

        if self.save_run then
            self:dumpRun()
        end

        return reward, terminal
    end

    -- TODO do something here
    if tc.state.game_ended then
        return
    end

    assert(tc.state.battle_frame_count % self.skip_frames == 0,
           "Frame count: " .. tc.state.battle_frame_count)

    if self.save_run then
        self:addToDump(a)
    end

    local reward, bterminal = self:getReward(a)

    local terminal = torch.zeros(1)
    if bterminal then
        self.battles_game = self.battles_game + 1
        if tc.state.battle_won then
            self.battles_won = self.battles_won + 1
        end
        terminal[1] = 1
        if self.save_run then
            self:dumpRun()
        end
    end

    self.battle_steps = self.battle_steps + 1

    return reward, terminal
end

function StarCraftMicro:suicideStep()
    -- get enemy unit
    local target
    for uid, ut in pairs(tc.state.units_enemy) do
        target = uid
        break
    end
    assert(target, "No target was found - units_enemy likely to be empty!")

    local actions = {}
    for uid, ut in pairs(tc.state.units_myself) do
        local command = tc.command(tc.command_unit_protected, uid,
                                   tc.cmd.Attack_Unit, target)
        table.insert(actions, command)
    end

    tc:send({table.concat(actions, ':')})

    local update = tc:receive()

    local terminal = false
    if tc.state.battle_just_ended then
        terminal = true
    end

    return terminal
end

function StarCraftMicro:getReward(a)
    local action = self:getTCActions(a)
    tc:send({table.concat(action, ':')})

    local update = tc:receive()

    -- Set types (must be all zeros at the start!)
    local reward = torch.zeros(1, self.opt.nagents):type(self.opt.dtype)
    local terminal = false

    if tc.state.battle_just_ended then
        terminal = true
    end

    self.last_a = a:clone()

    -- if it's first turn, reward & terminal == 0
    if self.previous_state_enemy ~= nil and self.previous_state_myself ~= nil then
        -- This function must return a scalar
        if self.opt.game_reward_function == 'battle' then
            local reward_all = self.reward_function()
            reward = reward + reward_all
        else
            reward = self.reward_function()
        end

    end

    self.previous_state_enemy = tablex.deepcopy(tc.state.units_enemy)
    self.previous_state_myself = tablex.deepcopy(tc.state.units_myself)

    return reward, terminal
end

function StarCraftMicro:getState(args)
    local state, full_state
    if self.opt.game_state_function == 'grid' then
        state = self.state_function(args)
        if self.opt.model_dueling == 1 or self.opt.model_remix_qs == 1 then
            full_state = self:getFullState()
        end
    elseif self.opt.game_state_function == 'action_range' then
        state = self.state_function(args)
        full_state = {table.unpack(state)} -- dirty copy hack?
    elseif self.opt.game_state_function == 'feat' then
        state, full_state = self.state_function(args)
    end
    return state, full_state
end

function StarCraftMicro:getStatistics()
    local stats = {}
    stats.battles_won = self.battles_won
    stats.battles_game = self.battles_game
    stats.win_rate = self.battles_won / (self.battles_game+1E-6)
    stats.timeouts = self.timeouts
    return stats
end

-- ----------------- STATE UTILS -----------------
function StarCraftMicro:update_theta(agent_id)

    local uid = self:mapIdtoTCId(agent_id,true)
    local ut = tc.state.units_myself[uid]

    if ut then

        -- if no action was taken in the previous step, then don't update theta
        local new_theta = self.theta[agent_id]

        -- if we attacked an enemy in previous step
        if self.last_a and self.previous_state_enemy_for_state and self.last_a[1][agent_id] > 5  and self.last_a[1][agent_id] ~= self.dead_action_ids  then

            local emapid = self.last_a[1][agent_id] - self.last_move_action_id
            local euid = self:mapIdtoTCId(emapid, false)

            local targ_pos = self.previous_state_enemy_for_state[euid].position
            local my_pos = tc.state.units_myself[uid].position
            local dy = targ_pos[2]-my_pos[2]
            local dx = targ_pos[1]-my_pos[1]

            --                 log.debugf("Agent %d on prev took action %d, attacking enemy %d  ",agent_id,self.last_a[1][agent_id],emapid)
            --                 log.debugf("target pos = %d %d, my pos = %d %d, diff = %d, %d", targ_pos[1],targ_pos[2], my_pos[1],my_pos[2], dx, dy)

            new_theta = math.deg(math.atan2( dy, dx )) % 360

            --                if  not ( ut.velocity[1]==0  and  ut.velocity[2]==0 ) then
            --                    log.errorf("Agent %d, vel = %s %s. When shooting vel should be 0",
            --                        agent_id,ut.velocity[1],ut.velocity[2])
            --                end


        elseif not ( ut.velocity[1]==0  and  ut.velocity[2]==0 ) then
            new_theta = math.deg(math.atan2(ut.velocity[2], ut.velocity[1])) % 360

        else
            --                if self.last_a and not ( self.last_a[1][agent_id]==1 or self.last_a[1][agent_id]==self.dead_action_ids) then
            --                    log.error(string.format('Agent = %d, action = %d, vel = %s, %s. If move action was taken then the velocity should be non-zero',
            --                        agent_id,self.last_a[1][agent_id],ut.velocity[1],ut.velocity[2]))
            --                end
        end

        self.theta[agent_id] = new_theta
    end

end

function StarCraftMicro:actionRangeState()
    local state = {}
    for agent_id = 1, self.opt.nagents do
        local action_range = self:getActionRange(nil, agent_id,true)[1][2]
        local m = torch.zeros(1,self.opt.game_action_space) - 1
        if self.opt.model_totally_blind == 0 then
            for k=1, #(action_range) do
                m[{1,action_range[k]}] = 1
                local temp = math.max(1, action_range[k]-4)
                if temp > 1 then
                    m[{1,action_range[k]}] = temp
                else
                    m[{1,action_range[k]}] = 0
                end
            end
        end
        state[agent_id] = {m}
    end
    return state
end

function StarCraftMicro:nGoodCriticFeatures()
    local nf_me_en = 1 --features of enemy in me-part (in-range, distance)
    local nf_me = 4 + self.opt.game_nagents --hp, cd, x, y, (shld, type)
    nf_me = nf_me + self.opt.game_action_space*(self.opt.model_critic_action_input + self.opt.model_critic_last_action_aware)
    local nf_en = 2
    if self.dz_map then
        nf_me = nf_me + 3
        nf_en = nf_en + 3
    end   
    local total_me_feats = nf_me + nf_me_en * self.opt.game_nenemies
    local total_agent_feats = total_me_feats + nf_en*self.opt.game_nenemies
    return total_me_feats, total_agent_feats, nf_en
end

function StarCraftMicro:goodCriticState()
    local ally_units = tc.state.units_myself
    local enemy_units = tc.state.units_enemy

    --don't change without changing self:nGoodCriticFeatures. sorry about this.
    local nf_me_en = 1 --features of enemy in me-part (in-range) (not distance anymore)
    local nf_me
    nf_me = 4 + self.opt.game_nagents --hp, cd, x, y, (shld, type)
    nf_me = nf_me + self.opt.game_action_space*(self.opt.model_critic_action_input + self.opt.model_critic_last_action_aware)
    local nf_en = 2 -- (distance, hp)
    if self.dz_map then
        nf_me = nf_me + 3
        nf_en = nf_en + 3
    end

    local center_y = 140
    local center_x = 85

    local default_range = nil
    if self.dz_map then
        default_range = 16
    end

    local all_agent_states = {}

    for i = 1, self.opt.game_nagents do
        local my_info = torch.zeros(nf_me + nf_me_en * self.opt.game_nenemies)
        local enemy_info = torch.zeros(nf_en * self.opt.game_nenemies)
        local me_uid = self:mapIdtoTCId(i,true)
        local me = ally_units[me_uid]
        if me then
            my_info[1] = me.hp / me.max_hp
            my_info[2] = me.gwcd / me.maxcd
            my_info[3] = (me.position[1] - center_x) / 50
            my_info[4] = (me.position[2] - center_y) / 50
            local idx = 4
            if self.dz_map then
                my_info[5] = me.shield / me.max_shield
                my_info[{{6,7}}] = self.one_hot_type:forward(me.type - 65 + 1)
                idx = 7
            end
            my_info[idx + i] = 1 -- one hot agent ID encoding
            idx = idx + self.opt.game_nagents + 1
            for j = 1,self.opt.game_nenemies do
                local uid = self:mapIdtoTCId(j,false)
                local features = enemy_units[uid]
                if features then
                    local me_idx = idx + (j-1)*nf_me_en
                    local en_idx = (j-1)*nf_en + 1
                    local enemy_x, enemy_y = unpack(features.position)
                    local distance = ((enemy_x - me.position[1])^2 + (enemy_y - me.position[2])^2)^0.5
                    local range = (default_range or math.max(16,me.gwrange)) * self.opt.game_state_fov_scale
                    my_info[me_idx] = (distance < range) and 1 or 0
                    -- my_info[me_idx+1] = distance / 50
                    enemy_info[en_idx] = distance / 50
                    enemy_info[en_idx+1] = features.hp / features.max_hp
                    if self.dz_map then
                        enemy_info[en_idx+2] = features.shield / features.max_shield
                        enemy_info[{{en_idx+3, en_idx+4}}] = self.one_hot_type:forward(features.type - 65 + 1)
                    end
                end
            end
        end
        local agent_state = torch.cat(my_info:view(1,-1), enemy_info:view(1,-1))
        table.insert(all_agent_states, agent_state)
        -- if global_state then
        --     global_state = torch.cat(global_state, agent_state)
        -- else
        --     global_state = agent_state:view(1,-1):clone()
        -- end
    end
    return all_agent_states
end

function StarCraftMicro:oldCriticState()
    local ally_units = tc.state.units_myself
    local enemy_units = tc.state.units_enemy

    local nf_al_cr, nf_en_cr
    nf_al_cr = 4
    nf_en_cr = 3

    if self.dz_map then
        nf_al_cr = nf_al_cr + 3
        nf_en_cr = nf_en_cr + 3
    end

    if self.opt.map == '2m_3zerg' or self.opt.map == '2w_4m' then
        nf_al_cr = nf_al_cr + 2
        nf_en_cr = nf_en_cr + 2
    end

    local global_state = torch.zeros(nf_al_cr*self.opt.game_nagents + nf_en_cr*self.opt.game_nenemies) - 1
    local center_y = 140
    local center_x = 85
    for i = 1, self.opt.game_nagents do
        local uid = self:mapIdtoTCId(i,true)
        if ally_units[uid] then
            local idx = nf_al_cr*(i-1)
            global_state[idx + 1] = ally_units[uid].hp / ally_units[uid].max_hp
            global_state[idx + 2] = (ally_units[uid].position[1] - center_x) / 50
            global_state[idx + 3] = (ally_units[uid].position[2] - center_y) / 50
            global_state[idx + 4] = ally_units[uid].gwcd / ally_units[uid].maxcd
            if self.dz_map then
                global_state[idx + 5] = ally_units[uid].shield / ally_units[uid].max_shield
                local type_id = ally_units[uid].type - 65 + 1
                global_state[{{idx + 6,idx + 7}}] = self.one_hot_type:forward(type_id)
            end
            if self.opt.map == '2m_3zerg' or self.opt.map == '2w_4m' then
                local type_id = self.unit_type_dict[ally_units[uid].type]
                global_state[{{idx + 5,idx + 6}}] = self.one_hot_type:forward(type_id)
            end
        end
    end
    for i = 1, self.opt.game_nenemies do
        local uid = self:mapIdtoTCId(i,false)
        if enemy_units[uid] then
            local idx = nf_en_cr*(i-1) + nf_al_cr*self.opt.game_nagents
            global_state[idx + 1] = enemy_units[uid].hp / enemy_units[uid].max_hp
            global_state[idx + 2] = (enemy_units[uid].position[1] - center_x) / 50
            global_state[idx + 3] = (enemy_units[uid].position[2] - center_y) / 50
            if self.dz_map then
                global_state[idx + 4] = enemy_units[uid].shield / enemy_units[uid].max_shield
                local type_id = enemy_units[uid].type - 65 + 1
                global_state[{{idx + 5,idx + 6}}] = self.one_hot_type:forward(type_id)
            end
            if self.opt.map == '2m_3zerg' or self.opt.map == '2w_4m' then
                local type_id = self.unit_type_dict[enemy_units[uid].type]
                global_state[{{idx + 4,idx + 5}}] = self.one_hot_type:forward(type_id)
            end
        end
    end
    return global_state
end

function StarCraftMicro:featState()
    local state = {}
    local ally_units = tc.state.units_myself
    local enemy_units = tc.state.units_enemy
    
    local nf_al_act, nf_en_act
    nf_al_act = 6
    nf_en_act = 5

    if self.dz_map then
        nf_al_act = nf_al_act + 3
        nf_en_act = nf_en_act + 3
    end

    if self.opt.map == '2m_3zerg' or self.opt.map == '2w_4m' then
        nf_al_act = nf_al_act + 2
        nf_en_act = nf_en_act + 2
    end

    if self.opt.game_state_limited_actor == 1 then
        nf_al_act = nf_al_act - 2 -- no health or cd
        nf_en_act = nf_en_act - 1 -- no health
    end

    local global_state
    if self.opt.model_good_critic == 1 or self.opt.use_newstate == 1 then
        global_state = self:goodCriticState()
    else
        global_state = self:oldCriticState()
    end

    for agent_id = 1, self.opt.nagents do
        local temp_state = {}
        temp_state.move_feats = torch.zeros(self.last_move_action_id)
        temp_state.enemy_feats = torch.zeros(self.opt.game_nenemies, nf_en_act) - 1 -- available, distance, hp, shield, type
        temp_state.ally_feats = torch.zeros(self.opt.game_nagents, nf_al_act) - 1 -- visible, distance, hp, cd, shield type
        -- could add distance as dx, dy; enemy cooldown.

        -- temp_state.ally_feats = torch.zeros(self.opt.game_nagents, 3) -- TODO. distance, hp, cd
        local uid = self:mapIdtoTCId(agent_id,true)
        if ally_units[uid] then -- otherwise dead, return HELLSTATE
            local my_x, my_y = unpack(ally_units[uid].position)

--            cds[{1,agent_id}] = ally_units[uid].awcd

            local m = torch.zeros(self.opt.game_action_space,1) - 1 -- -1:unavailable, 0:move, 1:available shoot
            local action_range = self:getActionRange(nil, agent_id,true)[1][2]
            for k=1, #(action_range) do
                if action_range[k] > self.last_move_action_id then
                    m[{action_range[k], 1}] = 1
                else
                    m[{action_range[k], 1}] = 0
                end
            end

            temp_state.move_feats = m[{{1,self.last_move_action_id}, {1}}]
            temp_state.enemy_feats = torch.repeatTensor(m[{{self.last_move_action_id + 1,self.opt.game_action_space - 1},{1}}],1,nf_en_act)

            for uid, features in pairs(enemy_units) do
                local mid = self:TCIdtoMapId(uid, false)
                --if attack possible (ie enemy observed), fill in info
                if temp_state.enemy_feats[{mid,1}] ~= -1 then
                    local enemy_x, enemy_y = unpack(features.position)
                    local distance = ((enemy_x - my_x)^2 + (enemy_y - my_y)^2)^0.5
                    temp_state.enemy_feats[{{mid},{2}}] = distance / 10
                    temp_state.enemy_feats[{{mid},{3}}] = (enemy_x - my_x) / 10
                    temp_state.enemy_feats[{{mid},{4}}] = (enemy_y - my_y) / 10
                    local idx = 5
                    if self.opt.game_state_limited_actor == 0 then
                        temp_state.enemy_feats[{{mid},{5}}] = features.hp / features.max_hp
                        idx = 6
                    end
                    if self.dz_map then
                        temp_state.enemy_feats[{{mid},{idx}}] = features.shield / features.max_shield
                        local type_id = features.type - 65 + 1
                        temp_state.enemy_feats[{{mid},{idx+1, idx+2}}] = self.one_hot_type:forward(type_id)
                    end
                    if self.opt.map == '2m_3zerg' or self.opt.map == '2w_4m' then
                        local type_id = self.unit_type_dict[features.type]
                        temp_state.enemy_feats[{{mid},{idx, idx+1}}] = self.one_hot_type:forward(type_id)
                    end
                end
            end

--            place the features of the agent himself always at the first place
            local idxs_map = {agent_id }
            for ii=1,self.opt.nagents do
                if ii ~= agent_id then
                    table.insert(idxs_map,ii)
                end
            end

            for stid, mid in pairs(idxs_map) do
--                local mid = self:TCIdtoMapId(uid, true)
                local uid = self:mapIdtoTCId(mid, true)
                local features = ally_units[uid]
                if features then
                    local ally_x, ally_y = unpack(features.position)
                    local distance = ((ally_x - my_x)^2 + (ally_y - my_y)^2)^0.5

                    local fov_scale = self.opt.game_state_fov_scale
                    local ally_in_range
                    if self.dz_map then
                        --                  hardcode the range to be the same for dragoons and zealots, equal to dragoon real range
                        local gwrange = 16
                        ally_in_range = distance <= gwrange*fov_scale
                    else
                        ally_in_range = distance <= math.max(16, ally_units[uid].gwrange)*fov_scale
                    end

                    if ally_in_range or self.opt.game_ally_fov == 0 then

                        temp_state.ally_feats[{{stid},{1}}] = 1
                        temp_state.ally_feats[{{stid},{2}}] = distance / 10
                        temp_state.ally_feats[{{stid},{3}}] = (ally_x - my_x) / 10
                        temp_state.ally_feats[{{stid},{4}}] = (ally_y - my_y) / 10
                        local idx = 5
                        if self.opt.game_state_limited_actor == 0 then
                            temp_state.ally_feats[{{stid},{5}}] = features.hp / features.max_hp
                            temp_state.ally_feats[{{stid},{6}}] = features.gwcd / features.maxcd
                            idx = 7
                        end
                        if self.dz_map then
                            temp_state.ally_feats[{{stid},{idx}}] = features.shield / features.max_shield
                            local type_id = features.type - 65 + 1
                            temp_state.ally_feats[{{stid},{idx + 1, idx + 2}}] = self.one_hot_type:forward(type_id)
                        end
                        if self.opt.map == '2m_3zerg' or self.opt.map == '2w_4m' then
                            local type_id = self.unit_type_dict[features.type]
                            temp_state.ally_feats[{{stid},{idx, idx+1}}] = self.one_hot_type:forward(type_id)
                        end
                    end
                end
            end

        end

        state[agent_id] = {torch.cat(temp_state.move_feats:view(1,-1), temp_state.enemy_feats:view(1,-1)):cat(temp_state.ally_feats:view(1,-1))}
    end

--    log.infof("Cwd = %s",cds)

    local full_state = {}
    for i=1,self.opt.game_nagents do
        if self.opt.use_newstate == 1 and self.opt.na_heads == 1 then
            local shuffle_state = global_state[i]
            for j =1,self.opt.game_nagents do
                if i ~= j then
                    shuffle_state = torch.cat(shuffle_state, global_state[j])
                end
            end
            full_state[i] = {shuffle_state:view(1,-1)}
        elseif (self.opt.use_newstate == 1 and self.opt.na_heads == 0) then
            local no_shuffle_state = global_state[1]
            for j = 2,self.opt.game_nagents do
                no_shuffle_state = torch.cat(no_shuffle_state, global_state[j])
            end
            full_state[i] = {no_shuffle_state:view(1,-1)}
        else
            if self.opt.na_heads == 0 then
                if self.opt.game_state_all_obs == 1 then
                    local cat_state = global_state:view(1,-1):clone()
                    for j=1,self.opt.game_nagents do
                        cat_state = torch.cat(cat_state, state[j][1])
                    end
                    full_state[i] = {cat_state:view(1,-1):clone()}
                else
                    -- full_state[i] = {torch.cat(global_state:view(1,-1),state[i][1])}
                    full_state[i] = {global_state:view(1,-1)}
                end
            else
                if self.opt.game_state_all_obs == 1 then
                    local cat_state = torch.cat(global_state:view(1,-1),state[i][1])
                    for j = 1, self.opt.game_nagents do
                        if j~= i then
                            cat_state = torch.cat(cat_state, state[j][1])
                        end
                    end
                    full_state[i] = {cat_state:view(1,-1):clone()}
                else
                    full_state[i] = {torch.cat(global_state:view(1,-1),state[i][1])}
                end
            end
        end
    end
    return state, full_state
end

-- TODO make featurizers
function StarCraftMicro:gridState()
    local state = {}

    local health = torch.FloatTensor()
    local full_grid

    if self.opt.game_state_health_own_all == 1 then
        health = torch.cat(health, self:getHealthTensor(true))
    end
    if self.opt.game_state_health_enemy  == 1 then
        health = torch.cat(health, self:getHealthTensor(false))
    end

--    local cds = torch.ones(1,self.opt.nagents)*-1
--    local velsx = torch.ones(1,self.opt.nagents)*-1
--    local velsy = torch.ones(1,self.opt.nagents)*-1

    for agent_id = 1, self.opt.nagents do
        local uid = self:mapIdtoTCId(agent_id,true)
        local agent_state = self:retrieveAgentGrids(agent_id)

        local temp_health = health:clone()
        if self.opt.game_state_health_own == 1 then
            local uhp
            if tc.state.units_myself[uid] then
                uhp = torch.ones(1)*tc.state.units_myself[uid].hp
            else
                uhp = torch.zeros(1) - 1
            end
            temp_health = torch.cat(health, uhp)
        end

        local ut = tc.state.units_myself[uid]

--        self:update_theta(agent_id)
        if ut then
            agent_state.cooldown = torch.ones(1)*ut.awcd
            agent_state.theta = torch.ones(1)*self.theta[agent_id]
--            cds[{1,agent_id}] = ut.awcd
--            velsx[{1,agent_id}] = ut.velocity[1]
--            velsy[{1,agent_id}] = ut.velocity[2]
        else
            agent_state.cooldown = torch.zeros(1)-1
            agent_state.theta = torch.zeros(1)-1
        end

        agent_state.health = temp_health
        state[agent_id] = agent_state
    end

    self.previous_state_enemy_for_state = tablex.deepcopy(tc.state.units_enemy)

--    log.tracef("Cwd = %s",cds)
    log.debug("Myself health = %s",torch.view(self:getHealthTensor(true),1,-1))
    log.debug("Enemy health = %s",torch.view(self:getHealthTensor(false),1,-1))
--    log.tracef("Velsx = %s",velsx)
--    log.tracef("Velsy = %s",velsy)
--    log.tracef("Thetas = %s",torch.reshape(self.theta,1,self.theta:size(1)) )


    return state
end

function StarCraftMicro:getFullState()
    -- gridposition health
    -- grid position ids
    local allied_grid, xy_allies = self:generateGridPosition(-1, true, nil) -- nil or "use_health"
    local enemy_grid, xy_enemies = self:generateGridPosition(-1, false, nil)

    -- local health_grid = health_state_enemies + health_state_allies
    -- local full_state = {health_grid=health_grid}

    full_state =  {grid_allies=allied_grid + 2,
                    xy_allies=xy_allies,
                    grid_enemies=enemy_grid + 2,
                    xy_enemies=xy_enemies}

    return full_state
end

function StarCraftMicro:retrieveAgentGrids(agent_id)

    local allied_grid = torch.ones(self.opt.game_state_grid_coarseness_height,
                                   self.opt.game_state_grid_coarseness_width) * 1
    local enemy_grid = torch.ones(self.opt.game_state_grid_coarseness_height,
                                  self.opt.game_state_grid_coarseness_width) * 1

    local uid = self:mapIdtoTCId(agent_id, true)

    local mode
    if self.opt.game_state_single_grid == 1 then
        mode = "use_grid_value"
    end
    if tc.state.units_myself[uid] ~= nil then
        -- unit is alive
        allied_grid, xy_allies = self:generateGridPosition(uid, true, nil)
        enemy_grid, xy_enemies = self:generateGridPosition(uid, false, mode)
         --offset for background encoding
        allied_grid = allied_grid + 2
        enemy_grid = enemy_grid + 2
        if self.opt.game_state_single_grid == 1 then
            allied_grid = allied_grid + enemy_grid
        end
    end


    if self.opt.game_state_single_grid == 0 then
        -- allied_grid[allied_grid:lt(0)] = 2
        -- enemy_grid[enemy_grid:lt(0)] = 2 -- traversible
        -- allied_grid[allied_grid:eq(0)] = 7 --blocked
        -- enemy_grid[enemy_grid:eq(0)] = 7
        return {grid_allies=allied_grid,
                xy_allies=xy_allies,
                grid_enemies=enemy_grid,
                xy_enemies=xy_enemies}
    else
        -- allied_grid[allied_grid:lt(0)] = 1
        -- allied_grid[allied_grid:eq(0)] = 2
        return {grid_allies=allied_grid,
                xy_allies=xy_allies}
    end
end

-- myself :: bool
function StarCraftMicro:generateGridPosition(cuid, myself, mode)
    local state, map, lookup_state, single_grid_value
    state = tc.state.units_myself
    if myself then
        lookup_state = state
        map = self.units_myself_map
        single_grid_value = 10
    else
        lookup_state = tc.state.units_enemy
        map = self.units_enemy_map
        single_grid_value = 6
    end

    local grid, c_width, c_height, width, height
    if cuid == -1 then
        c_height = self.opt.game_state_full_state_grid_coarseness_height
        c_width = self.opt.game_state_full_state_grid_coarseness_width
        height = self.opt.game_state_full_state_grid_height
        width = self.opt.game_state_full_state_grid_width
    else
        c_height = self.opt.game_state_grid_coarseness_height
        c_width = self.opt.game_state_grid_coarseness_width
        height = self.opt.game_state_grid_height
        width = self.opt.game_state_grid_width
    end

    grid = torch.zeros(c_height,
                       c_width)

    local unit, center_x, center_y

    if cuid == -1 then
        center_y = 140
        center_x = 85
    else
        unit = state[cuid]
        center_x = unit.position[1]
        center_y = unit.position[2]
    end

    -- TODO cleanup
    if self.opt.game_state_grid_with_boundaries == 1 then
        -- north
        local visible_dead = self.state_boundaries.north - center_y - 10
        local amount = (c_height / 2)
            - math.abs(visible_dead / self.opt.game_state_grid_expected_unit_size)
        for i = 1, torch.round(amount) do
            grid[i] = -1
        end

        -- south
        local visible_dead = self.state_boundaries.south - center_y + 10
        local amount = (c_height / 2)
            - math.abs(visible_dead / self.opt.game_state_grid_expected_unit_size)
        for i = 1, torch.round(amount) do
            grid[c_height - i + 1] = -1
        end

        -- west
        local visible_dead = self.state_boundaries.west - center_x - 10
        local amount = (c_width / 2)
            - math.abs(visible_dead / self.opt.game_state_grid_expected_unit_size)
        local y
        for i = 1, torch.round(amount) do
            y = grid:select(2, i)
            y:fill(-1)
        end

        -- east
        local visible_dead = self.state_boundaries.east - center_x + 10
        local amount = (c_width / 2)
            - math.abs(visible_dead / self.opt.game_state_grid_expected_unit_size)
        local y
        for i = 1, torch.round(amount) do
            y = grid:select(2, c_width - i + 1)
            y:fill(-1)
        end
    end

    local dx, dy
    local xy_locs = {}
    for uid, ut in pairs(lookup_state) do
        dx = math.floor((ut.position[1] - center_x + width / 2)
                / self.grid_elem_size_width) + 1
        dy = math.floor((ut.position[2] - center_y + height / 2)
                / self.grid_elem_size_height) + 1
        if cuid ~= uid or self.opt.game_state_grid_add_center == 1 then
            if (dx > 0 and dx <= c_width
                and dy > 0 and dy <= c_height) then
                local mid
                if not mode then
                    mid = self:TCIdtoMapId(uid, myself)
                elseif mode == "use_health" then
                    mid = lookup_state[uid].hp
                    -- Return negative health for enemies
                    if not myself then
                        mid = mid * -1
                    end
                elseif mode == "use_grid_value" then
                    mid = single_grid_value
                end

                if myself and self.opt.game_state_forever_alone == 1 then
                    grid[grid:size()[1]][self:TCIdtoMapId(uid, myself)] = self:TCIdtoMapId(uid, myself)
                else
                    grid[dy][dx] = mid or 0
                    xy_locs[mid] = {dy, dx}
                end
            end
        end
    end
    return grid, xy_locs
end

function StarCraftMicro:getHealthTensor(myself)
    local state, map, n_units
    if myself then
        state = tc.state.units_myself
        map = self.units_myself_map
        n_units = self.opt.nagents
    else
        state = tc.state.units_enemy
        map = self.units_enemy_map
        n_units = self.opt.nenemies
    end

    local ut
    local hpt = torch.zeros(n_units)
    for uid, u in pairs(map) do
        ut = self:getUnitById(state, u.tc_id)
        if not ut then
            hpt[uid] = -1
        else
            hpt[uid] = ut.hp
        end
    end
    return hpt
end

function StarCraftMicro:isUnitAlive(id)
    local uid = self:mapIdtoTCId(id, true)
    if tc.state.units_myself[uid] then
        return true
    else
        return false
    end
end


-- -------------------- ACTION SPACE ---------------------

-- TODO parametrise this by number of enemies!!
function StarCraftMicro:getTCActions(a)
    -- a should be a tensor / table with one index per agent
    local actions = {}
    local command

    -- TODO change this once we have cleaned up the index madness in comm.lua
    for i=1,a:size()[2] do
        local uid = self:mapIdtoTCId(i, true)
        command = self.action_function(a[1][i], uid)
        table.insert(actions, command)
    end

    if self.opt.bait == 1 then
        for i, unit in pairs(self.units_bait) do
            if (self.tc.state.units_myself[unit]) then
                if tablex.find(self:getActionRange(nil, -1)[1][2], self.bait_random_action) then
                    command = self.action_function(self.bait_random_action, unit)
                else
                    command = self.action_function(1, unit)
                end
                table.insert(actions, command)
            end
        end
    end

    return actions
end

function StarCraftMicro:relativeAttack(index, uid)
    -- Need this for assert
    -- UPDATE ME anytime action mapping changes
    local first_attack_action_id = 6

    assert(first_attack_action_id == self.last_move_action_id + 1)

    local position, target, command

    local ut = self:getUnitById(tc.state.units_myself, uid)

    if ut == nil then
        return
    end

    if index == 1 then
        if self.opt.game_no_unit_stop == 1 then
            command = nil
        else
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Stop)
        end
    elseif index == 2 then
        -- move north
        if ut.position[2] > self.state_boundaries.north then
            position = {ut.position[1],
                        ut.position[2] - self.opt.game_action_move_amount}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
    elseif index == 3 then
        -- move south
        if ut.position[2] < self.state_boundaries.south then
            position = {ut.position[1],
                        ut.position[2] + self.opt.game_action_move_amount}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
    elseif index == 4 then
        -- move west
        if ut.position[1] > self.state_boundaries.west then
            position = {ut.position[1] - self.opt.game_action_move_amount,
                        ut.position[2]}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
    elseif index == self.last_move_action_id then
        -- move east
        if ut.position[1] < self.state_boundaries.east then
            position = {ut.position[1] + self.opt.game_action_move_amount,
                        ut.position[2]}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
        -- ATTACK
    elseif index < self.dead_action_ids then
        local action_index = index - self.last_move_action_id
        local targets = self:getRelativeTargets(uid, action_index)
        local min = 100000
        local target
        if #targets > 0 then
            for i=1, #targets do
                if targets[i].dist < min then
                    min = targets[i].dist
                    target = targets[i].uid
                end
            end
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Attack_Unit,
                                 target)
        end
    end
    assert(index <= self.dead_action_ids and index > 0, "Action index is invalid! Got " .. index)
    if index < self.dead_action_ids then
        if self.opt.game_no_unit_stop == 0 and self.opt.game_demo == 0 then
            assert(not(command == nil), "Got nill command with index: " .. index)
        else
            assert(not(command == nil)
                       or (command == nil
                               and index == 0),
                   "Got nill command with index: " .. index)
        end
    end
    return command
end

-- This method expects indices 1 to max_attack
function StarCraftMicro:getRelativeTargets(cuid, index)
    assert(index > 0 and index <= self.opt.n_relative_attack_sections)
    local cut = self.tc.state.units_myself[cuid]
    assert(cut)
    local radial_step = 360 / self.opt.n_relative_attack_sections
    local lower_bound, upper_bound
    if index == 1 then
        lower_bound = 0
    else
        lower_bound = radial_step * (index - 1)
    end
    upper_bound = (radial_step * index)

    local targets = {}
    for uid, ut in pairs(self.tc.state.units_enemy) do
        local diff_x = ut.position[1] - cut.position[1]
        local diff_y = ut.position[2] - cut.position[2]
        local dist = math.sqrt(diff_x^2 + diff_y^2)
        local angle = math.deg(math.atan2(diff_x, diff_y)) % 360
        -- print(dist, "|", cut.gwrange, "|", angle, "|", upper_bound, "|",
        --       lower_bound)

        -- Not <= gwrange because don't want to introduce
        --       floating points errors and additional hidden dynamics.
        if (dist < cut.gwrange
                and angle < upper_bound
            and angle >= lower_bound) then
            local unit = {
                ut = ut,
                uid = uid,
                dist = dist,
                angle = angle
            }
            table.insert(targets, unit)
        end
    end
    return targets
end

-- TODO this index might need to be index - 1 because tensor
-- can contain zeros
function StarCraftMicro:indexAttack(index, uid)
    -- Need this for assert
    -- UPDATE ME anytime action mapping changes
    local first_attack_action_id = 6

    assert(first_attack_action_id == self.last_move_action_id + 1)

    local position, command

    local ut = self:getUnitById(tc.state.units_myself, uid)

    if ut == nil then
        return
    end

    if index == 1 then
        if self.opt.game_no_unit_stop == 1 then
            command = nil
        else
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Stop)
        end
    elseif index == 2 then
        -- move north
        if ut.position[2] > self.state_boundaries.north then
            position = {ut.position[1],
                        ut.position[2] - self.opt.game_action_move_amount}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
    elseif index == 3 then
        -- move south
        if ut.position[2] < self.state_boundaries.south then
            position = {ut.position[1],
                        ut.position[2] + self.opt.game_action_move_amount}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
    elseif index == 4 then
        -- move west
        if ut.position[1] > self.state_boundaries.west then
            position = {ut.position[1] - self.opt.game_action_move_amount,
                        ut.position[2]}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
    elseif index == self.last_move_action_id then
        -- move east
        if ut.position[1] < self.state_boundaries.east then
            position = {ut.position[1] + self.opt.game_action_move_amount,
                        ut.position[2]}
            command = tc.command(tc.command_unit_protected, uid,
                                 tc.cmd.Move, -1,
                                 position[1], position[2])
        end
    elseif index < self.dead_action_ids then
        command = tc.command(tc.command_unit_protected, uid,
                             tc.cmd.Attack_Unit, self:mapIdtoTCId(index - self.last_move_action_id, false))
    end

    local agent_id = self:TCIdtoMapId(uid, true)

    --  this is needed for the individual rewards calculation. We make it 1.1 so we dont miss a reward ( it is better risking getting some more than we should)
    local effective_fov_scale = 1.1
    local effective_action_range = self:getActionRange(nil,agent_id,true,effective_fov_scale)[1][2]
    self.last_effective_action_range[agent_id] = effective_action_range

    if self.opt.game_disable_action_range==1 then
    --    print(torch.view(torch.Tensor(action_range),1,-1))
    --    print("agent", agent_id, "index", index, "hasvalue", util.has_value(action_range, index))
        local action_range = self:getActionRange(nil,agent_id,true)[1][2]
        if not util.has_value(action_range, index) then
            command = tc.command(tc.command_unit_protected, uid, tc.cmd.Stop)
        end
    end

    assert(index <= self.dead_action_ids and index > 0, "Action index is invalid! Got " .. index)
    if index < self.dead_action_ids and self.opt.game_demo == 0 then
        if self.opt.game_no_unit_stop == 0 then
            assert(not(command == nil), "Got nill command with index: " .. index)
        else
            assert(not(command == nil) or (command == nil and index == 0), "Got nill command with index: " .. index)
        end
    end

    return command
end


-- --------------------- ID utils --------------------

-- if not found returns nil
function StarCraftMicro:getUnitById(units, id)
    for uid, ut in pairs(units) do
        if uid == id then
            return ut
        end
    end
end

function StarCraftMicro:mapUnits(some_units, with_bait)
    local units = {}
    local units_map = {}

    function sortByY(a,b) return a[2] < b[2] end
    function sortByX(a,b) return a[3] < b[3] end
    function sortByType(a,b) return a[4] < b[4] end

    -- FIXME this is more stochastic than we would like
    for uid, ut in pairs(some_units) do
        table.insert(units, {uid, ut.position[2], ut.position[1], ut.type})
    end


    if with_bait and self.opt.bait == 1 then
        self.units_bait = {}
        table.sort(units, sortByX)
        local ub = units[#units]
        if ub then
            table.insert(self.units_bait, ub[1])
            units = tablex.removevalues(units, #units, #units)
        end
    end

    table.sort(units, sortByY)
    table.sort(units, sortByType)

    local id = 1

    for uid, ut in pairs(units) do
        local m = {}
        m.tc_id = ut[1]
        m.id = id
        table.insert(units_map, m)
        id = id + 1
    end

    return units_map
end

function StarCraftMicro:mapIdtoTCId(id, myself)
    local state, map
    if myself then
        state = tc.state.units_myself
        map = self.units_myself_map
    else
        state = tc.state.units_enemy
        map = self.units_enemy_map
    end

    for i, u in pairs(map) do
        if u.id == id then
            return u.tc_id
        end
    end
end

function StarCraftMicro:TCIdtoMapId(id, myself)
    local state, map
    if myself then
        state = tc.state.units_myself
        map = self.units_myself_map
    else
        state = tc.state.units_enemy
        map = self.units_enemy_map
    end
    for i, u in pairs(map) do
        if u.tc_id == id then
            return u.id
        end
    end
end


-- --------------------- REWARD FUNCTIONS ----------------------

function StarCraftMicro:rewardBattle()
    -- where reward =
    --  delta health - delta enemies + delta deaths where value:
    --   if enemy unit dies, add game_reward_death_value per dead unit
    --   if own unit dies, subtract game_reward_death_value per dead unit

    local delta_deaths = 0
    local delta_myself = 0
    local delta_enemy = 0

    -- update deaths
    for i = 1, self.opt.nagents do
        -- my units
        if self.death_tracker_myself[i] == 0 and self:mapIdtoTCId(i, true) then
            local tcid = self:mapIdtoTCId(i, true)
            local prev_hp = self.previous_state_myself[tcid].hp + self.previous_state_myself[tcid].shield * self.opt.game_reward_shield
            if not tc.state.units_myself[tcid] then
                -- just died
                self.death_tracker_myself[i] = 1
                if self.opt.game_reward_only_positive == 1 then
                    delta_deaths = delta_deaths
                else
                    delta_deaths = delta_deaths - self.opt.game_reward_death_value*self.opt.game_reward_negative_scale
                end
                delta_myself = delta_myself + self.opt.game_reward_negative_scale*prev_hp
            else
                delta_myself = delta_myself + self.opt.game_reward_negative_scale*(prev_hp - tc.state.units_myself[tcid].hp - tc.state.units_myself[tcid].shield * self.opt.game_reward_shield)
            end
        end
    end

    for i = 1, self.opt.nenemies do
        -- enemy units
        if self.death_tracker_enemy[i] == 0 and self:mapIdtoTCId(i, false) then
            local tcid = self:mapIdtoTCId(i, false)
            local prev_hp = self.previous_state_enemy[tcid].hp + self.previous_state_enemy[tcid].shield * self.opt.game_reward_shield
            if not tc.state.units_enemy[tcid] then
                self.death_tracker_enemy[i] = 1
                delta_deaths = delta_deaths + self.opt.game_reward_death_value
                delta_enemy = delta_enemy + prev_hp*self.opt.game_reward_dhp_weight
            else
                delta_enemy = delta_enemy + (prev_hp - tc.state.units_enemy[tcid].hp - tc.state.units_enemy[tcid].shield * self.opt.game_reward_shield)*self.opt.game_reward_dhp_weight
            end
        end
    end

    local reward = 0

    if self.opt.game_reward_only_positive == 1 then
        reward = delta_enemy + delta_deaths
    else
        reward = delta_enemy + delta_deaths - delta_myself + self.opt.game_reward_step_value
    end

    if self.tc.state.battle_just_ended and self.tc.state.battle_won then
        reward = reward + self.opt.game_reward_win
    end

    return reward

end

function StarCraftMicro:rewardIndividual()
    -- where reward =
    --  delta health[i] - delta enemy_i where enemy_i the enemy that the unit attacked, if any:

    local damage_myself = torch.zeros(self.opt.nagents):type(self.opt.dtype)
    local damage_enemy = torch.zeros(self.opt.nagents):type(self.opt.dtype)
    local death_reward_enemy = torch.zeros(self.opt.nagents):type(self.opt.dtype)
    local delta_enemy = 0

    local spikes = torch.zeros(self.opt.nagents):type(self.opt.dtype)
    local leg_ats = torch.zeros(self.opt.nagents):type(self.opt.dtype)
    local damage_per_enemy = torch.zeros(self.opt.nenemies):type(self.opt.dtype)
    local just_died_enemy = torch.zeros(self.opt.nenemies):type(self.opt.dtype)

    local just_died_myself = torch.zeros(self.opt.nagents):type(self.opt.dtype)

    assert (self.last_a,"Last action hasn't been set - shomething's wrong")

    local spk = false

    -- update deaths
    for i = 1, self.opt.nagents do
        -- my units
        if self.death_tracker_myself[i] == 0 and self:mapIdtoTCId(i, true) then
            local tcid = self:mapIdtoTCId(i, true)
            local prev_hp = self.previous_state_myself[tcid].hp + self.previous_state_myself[tcid].shield * self.opt.game_reward_shield

            local cur_cooldown

            local prev_cooldown = self.previous_state_myself[tcid].awcd

            local cd_ok, just_shot
            cd_ok = (prev_cooldown-self.skip_frames)<=1

            if not tc.state.units_myself[tcid] then
                -- just died
                self.death_tracker_myself[i] = 1
                just_died_myself[i] = 1
                damage_myself[i] = prev_hp + self.opt.game_reward_death_value
            else
                damage_myself[i] = (prev_hp - tc.state.units_myself[tcid].hp - tc.state.units_myself[tcid].shield * self.opt.game_reward_shield)
                cur_cooldown = tc.state.units_myself[tcid].awcd
                just_shot = (cur_cooldown-self.skip_frames)>1
            end

            local agent_action = self.last_a[1][i]

            local is_attack_act = (agent_action > 5 and agent_action ~= self.dead_action_ids)

            if cd_ok and just_shot then
--                log.warnf("Just spiked agent %d, action %d, cd %d",i,agent_action, cur_cooldown); --print(enemies_shot_by_agents)
                spk = true
                self.n_spikes = self.n_spikes + 1
                spikes[i] = 1
            end

            if cd_ok and is_attack_act then
                local emapid = agent_action - self.last_move_action_id
--                log.warnf("Enemy %d shot by agent %d",emapid,i);
                self.n_legataks = self.n_legataks + 1
                leg_ats[i] = emapid
            end

            if cd_ok and just_shot and is_attack_act then
--                log.errorf("Agent's %d cooldown is up but no shoot action",i); --print(enemies_shot_by_agents)
            end

        end
    end

    for i = 1, self.opt.nenemies do
        -- enemy units
        if self.death_tracker_enemy[i] == 0 and self:mapIdtoTCId(i, false) then
            local tcid = self:mapIdtoTCId(i, false)
            local prev_hp = self.previous_state_enemy[tcid].hp + self.previous_state_enemy[tcid].shield * self.opt.game_reward_shield

            local dmg = 0
            if not tc.state.units_enemy[tcid] then
                self.death_tracker_enemy[i] = 1
                dmg = prev_hp
                just_died_enemy[i] = 1
            else
                dmg = (prev_hp - tc.state.units_enemy[tcid].hp - tc.state.units_enemy[tcid].shield * self.opt.game_reward_shield)
            end
            damage_per_enemy[i] = dmg
        end
    end

    local reward_battle_won = nil
    if self.tc.state.battle_just_ended then
        if self.tc.state.battle_won then
            reward_battle_won =  self.opt.game_reward_win
        else
            reward_battle_won =  0
        end
    end

    return {spikes=spikes, leg_ats = leg_ats, dhp=damage_per_enemy,
        just_died_enemy=just_died_enemy, death_tracker_myself=self.death_tracker_myself:clone(),
        reward_battle_won = reward_battle_won
    }

end

function StarCraftMicro:dumpRun()
    local s = 'dump_run_step_' .. string.format("%.4d", self.battles_game) .. '.csv'
    csvigo.save({
            path = paths.concat("./results/dumps/", s),
            data = self.current_dump,
            verbose = false
    })
    self:createDump()
end

function StarCraftMicro:createDump()
    local dump = {"step"}

    for i=1, self.opt.nagents do
        table.insert(dump, "ax" .. i)
        table.insert(dump, "ay" .. i)
    end

    self.current_dump = {dump}

end
function StarCraftMicro:addToDump(a)
    local at = {string.format("%.2d", self.battle_steps)}

    for i=1,a:size()[2] do
        local uid = self:mapIdtoTCId(i, true)
        local u = self.tc.state.units_myself[uid]
        if u then
            table.insert(at, u.position[1])
            table.insert(at, u.position[2])
        else
            table.insert(at, -1)
            table.insert(at, -1)
        end
    end
    table.insert(self.current_dump, at)
end

return StarCraftMicro
