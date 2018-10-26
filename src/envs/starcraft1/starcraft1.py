from ..multiagentenv import MultiAgentEnv
from .map_params import get_map_params, map_present

try:
    import torchcraft as tc
    import torchcraft.Constants as tcc
except:
    pass

import numpy as np
import sys
import os
import subprocess
import math
import time
from operator import attrgetter
from copy import deepcopy
import portpicker
import socket
import errno

from utils.dict2namedtuple import convert

'''
StarCraft I: Brood War
'''


class SC1(MultiAgentEnv):

    def __init__(self, **kwargs):

        self.debug_launcher = False
        self.port_in_use = False
        self.debug_inputs = False
        self.debug_rewards = False

        if self.debug_launcher:
            print("INIT")

        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        self._add_deepcopy_support()

        # Read arguments
        self.map_name = args.map_name
        assert map_present(self.map_name), \
            "map {} not in map registry! please add.".format(self.map_name)
        map_params = convert(get_map_params(self.map_name))
        self.map_type = map_params.map_type
        self.n_agents = map_params.n_agents
        self.n_enemies = map_params.n_enemies
        self._agent_race = map_params.agent_race
        self._bot_race = map_params.bot_race
        self.zealot_id = 65
        self.dragoon_id = 66
        self.episode_limit = map_params.limit
        self.micro_battles = map_params.micro_battles

        self._move_amount = args.move_amount
        self._step_mul = args.step_mul
        self.state_last_action = args.state_last_action

        # Rewards args
        self.reward_only_positive = args.reward_only_positive
        self.reward_negative_scale = args.reward_negative_scale
        self.reward_death_value = args.reward_death_value
        self.reward_win = args.reward_win
        self.reward_scale = args.reward_scale
        self.reward_scale_rate = args.reward_scale_rate

        # Other
        self.seed = args.seed
        self.heuristic = args.heuristic
        self.measure_fps = args.measure_fps
        self.continuing_episode = args.continuing_episode

        self.hostname = args.hostname
        self.port = portpicker.pick_unused_port()

        self.n_actions_no_attack = 6
        self.n_actions = self.n_actions_no_attack + self.n_enemies
        self.max_reward = self.n_enemies * self.reward_death_value + self.reward_win

        for tc_dir in ["/install/torchcraft"]:
            if os.path.isdir(tc_dir):
                os.environ['TCPATH'] = tc_dir

        if sys.platform == 'linux':
            os.environ['SC1PATH'] = os.path.join(os.getcwd(), '3rdparty', 'StarCraftI', 'linux')
            self.env_file_type = 'so'
        elif sys.platform == 'darwin':
            os.environ['SC1PATH'] = os.path.join(os.getcwd(), '3rdparty', 'StarCraftI', 'mac')
            self.env_file_type = 'dylib'

        # Check if server has already been launched on this port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((socket.gethostbyname(socket.gethostname()), self.port))
            self.port_in_use = False
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                # Port is already in use
                self.port_in_use = True
                print("Exception error: Port {} already in use. \n".format(self.port, e))
            else:
                # Something else raised the socket.error exception
                print(e)
        s.close()

        # For single-batch testing in BWAPILauncher rendering
        # self.port = 11111  # Needs to be commented out when using more than once SC1 instance

        if self.debug_launcher:
            print("BEFORE LAUNCH SERVER")

        # Launch the server
        if not self.port_in_use:
            self._launch_server()

        if self.debug_launcher:
            print("BEFORE LAUNCH CLIENT")

        # Launch the game
        self._launch_client()

        if self.debug_launcher:
            print("AFTER LAUNCH CLIENT")

        self.map_x = self._obs.map_size[0]
        self.map_y = self._obs.map_size[1]
        self.map_play_area_min = [int(0), int(0)]
        self.map_play_area_max = [self.map_x, self.map_y]
        self.max_distance_x = self.map_x
        self.max_distance_y = self.map_y

        self._episode_count = -1
        self._total_steps = 0

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0

    def _launch_server(self):
        os_env = os.environ.copy()
        bwenv_path = '{}/BWEnv/build/BWEnv.{}'.format(os.environ['TCPATH'], self.env_file_type)
        if not os.path.isfile(bwenv_path):
            bwenv_path = '{}/bots/BWEnv.{}'.format(os.environ['SC1PATH'], self.env_file_type)
        my_env = {"OPENBW_ENABLE_UI": '0',
                  "BWAPI_CONFIG_AI__RACE": '{}'.format(self._bot_race),
                  "BWAPI_CONFIG_AI__AI": bwenv_path,
                  "BWAPI_CONFIG_AUTO_MENU__AUTO_MENU": "SINGLE_PLAYER",
                  "BWAPI_CONFIG_AUTO_MENU__MAP": '{}/src/envs/starcraft1/maps/{}.{}'.format(os.getcwd(),
                                                                                            self.map_name,
                                                                                            self.map_type),
                  "BWAPI_CONFIG_AUTO_MENU__GAME_TYPE": "USE_MAP_SETTINGS",
                  "TORCHCRAFT_PORT": '{}'.format(self.port)}

        launcher_path = os.environ['SC1PATH']
        my_env = {**os_env, **my_env}
        launcher = 'BWAPILauncher'
        subprocess.Popen([launcher], cwd=launcher_path, env=my_env)

    def _launch_client(self):
        self.controller = tc.Client()
        self.controller.connect(self.hostname, self.port)
        self._obs = self.controller.init(micro_battles=self.micro_battles)

        self.controller.send([
            [tcc.set_combine_frames, self._step_mul],
            [tcc.set_speed, 0],
            [tcc.set_gui, 1],
            [tcc.set_cmd_optim, 1],
        ])

    def reset(self):
        """Start a new episode."""

        if self.debug_inputs or self.debug_rewards:
            print('------------>> RESET <<------------')

        self._episode_count += 1
        self._episode_steps = 0
        if self._episode_count > 0:  # no need to restart in the first episode
            self._restart()

        self._episode_count += 1

        # Naive way to measure FPS
        if self.measure_fps:
            if self._episode_count == 10:
                self.start_time = time.time()

            if self._episode_count == 20:
                elapsed_time = time.time() - self.start_time
                print('-------------------------------------------------------------')
                print("Took %.3f seconds for %s steps with step_mul=%d: %.3f fps" % (
                    elapsed_time, self._total_steps, self._step_mul,
                    (self._total_steps * self._step_mul) / elapsed_time))
                print('-------------------------------------------------------------')

        # Information kept for computing the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_agent_units = None
        self.previous_enemy_units = None

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self._obs = self.controller.recv()
        self.init_units()

        # self.full_restart()  # TODO: Implement condition to trigger full restart in case the client has crashed

        return self.get_obs(), self.get_state()

    def _restart(self):
        # End episode and start a new one by killing and restoring all units
        self.kill_all_units()
        self._obs = self.controller.init(micro_battles=self.micro_battles)

        # self.full_restart()  # TODO: Implement condition to trigger full restart in case the client has crashed

    def full_restart(self):
        # Close and relaunch client
        self.controller.close()
        self._launch_client()

        self.force_restarts += 1

        self.reset()

    def one_hot(self, data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def step(self, actions):
        """ Returns reward, terminated, info """
        self.last_action = self.one_hot(actions, self.n_actions)

        sc_actions = []
        for a_id, action in enumerate(actions):
            if not self.heuristic:
                agent_action = self.get_agent_action(a_id, action)
            else:
                agent_action = self.get_agent_action_heuristic(a_id, action)
            if agent_action:
                sc_actions.append(agent_action)

        # Send actions
        self.controller.send(sc_actions)
        self._obs = self.controller.recv()

        # self.full_restart()  # TODO: Implement condition to trigger full restart in case the client has crashed

        self._total_steps += 1
        self._episode_steps += 1

        # Update what we know about units
        game_ended = self.update_units()

        terminated = False
        if game_ended != 0:
            self.reward = self.reward_battle()
        info = {"battle_won": False}

        if game_ended is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_ended == 1:
                self.battles_won += 1
                info["battle_won"] = True
                self.reward += self.reward_win

        elif self.episode_limit > 0 and self._episode_steps >= self.episode_limit:
            # Episode limit has been reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug_inputs or self.debug_rewards:
            print("Total Reward = %.f \n ---------------------" % self.reward)

        if self.reward_scale:
            self.reward /= (self.max_reward / self.reward_scale_rate)

        return self.reward, terminated, info

    def get_agent_action(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        x = unit.x
        y = unit.y

        action = action.item()

        if action == 0:
            # no-op (valid only when dead)
            if unit.health > 0:
                print("break")
            try:
                assert unit.health == 0, "No-op chosen but the agent's unit is not dead"
            except Exception as e:
                print("episode_steps:", self._episode_steps)
                print("total_steps:", self._total_steps)
                print("obs:", self._obs)
                pass
            if self.debug_inputs:
                print("Agent %d: Dead" % a_id)
            return None

        elif action == 1:
            # stop
            sc_action = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Stop]
            if self.debug_inputs:
                print("Agent %d: Stop" % a_id)

        elif action == 2:
            # Move up
            sc_action = [tcc.command_unit_protected,
                         unit.id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x),
                         int(y + self._move_amount)]
            if self.debug_inputs:
                print("Agent %d: North" % a_id)

        elif action == 3:
            # Move down
            sc_action = [tcc.command_unit_protected,
                         unit.id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x),
                         int(y - self._move_amount)]
            if self.debug_inputs:
                print("Agent %d: South" % a_id)

        elif action == 4:
            # Move right
            sc_action = [tcc.command_unit_protected,
                         unit.id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x + self._move_amount),
                         int(y)]
            if self.debug_inputs:
                print("Agent %d: East" % a_id)

        elif action == 5:
            # Move left
            sc_action = [tcc.command_unit_protected,
                         unit.id,
                         tcc.unitcommandtypes.Move,
                         -1,
                         int(x - self._move_amount),
                         int(y)]
            if self.debug_inputs:
                print("Agent %d: West" % a_id)
        else:
            # Attack units that are in shooting range
            enemy_id = action - self.n_actions_no_attack
            enemy_unit = self.enemies[enemy_id]
            sc_action = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Attack_Unit, enemy_unit.id]
            if self.debug_inputs:
                print("Agent %d attacks enemy # %d" % (a_id, enemy_id))

        return sc_action

    def get_agent_action_heuristic(self, a_id, action):

        # Heuristic to attack closest enemy unit in shooting range
        unit = self.get_unit_by_id(a_id)
        shooting_range = self.unit_shoot_range(a_id)
        dist = float('inf')
        enemy_id = None

        # Find the closest enemy unit that's alive and in shooting range
        for e_id, e_unit in self.enemies.items():
            d = ((e_unit.x - unit.x) ** 2 + (e_unit.y - unit.y) ** 2) ** 0.5
            if d < dist and d < shooting_range and e_unit.health > 0:
                dist = d
                enemy_id = e_id

        # Attack closest enemy unit or stop if no enemies are in shooting range
        if enemy_id is not None:
            sc_action = [tcc.command_unit_protected, a_id, tcc.unitcommandtypes.Attack_Unit, enemy_id]
            if self.debug_inputs:
                print("Agent %d attacks enemy # %d" % (a_id, enemy_id))
        else:
            sc_action = [tcc.command_unit_protected, a_id, tcc.unitcommandtypes.Stop]
            if self.debug_inputs:
                print("Agent %d: Stop" % a_id)

        return sc_action

    def reward_battle(self):
        #  delta health - delta enemies + delta deaths where value:
        #    If enemy unit dies, add reward_death_value per dead unit
        #    If own unit dies, subtract reward_death_value per dead unit

        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        if self.debug_rewards:
            for al_id in range(self.n_agents):
                print("Agent %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_agent_units[al_id].health \
                                                                      - self.agents[al_id].health,
                                                                      self.previous_agent_units[al_id].shield \
                                                                      - self.agents[al_id].shield))
            print('---------------------')
            for al_id in range(self.n_enemies):
                print("Enemy %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_enemy_units[al_id].health \
                                                                      - self.enemies[al_id].health,
                                                                      self.previous_enemy_units[al_id].shield \
                                                                      - self.enemies[al_id].shield))

        # Update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # Has not die so far
                prev_health = self.previous_agent_units[al_id].health + self.previous_agent_units[al_id].shield
                if al_unit.health == 0:
                    # Just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # Still alive
                    delta_ally += (prev_health - al_unit.health - al_unit.shield) * neg_scale

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = self.previous_enemy_units[e_id].health + self.previous_enemy_units[e_id].shield
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                # still alive
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:

            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Reward: ", delta_enemy + delta_deaths)
                print("--------------------------")

            reward = delta_enemy + delta_deaths

        else:
            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Delta ally: ", - delta_ally)
                print("Reward: ", delta_enemy + delta_deaths - delta_ally)
                print("--------------------------")

            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        shooting_range = unit.groundRange if unit.type != 65 else 16
        return shooting_range

    def unit_sight_range(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        sight_range = tcc.staticvalues["sightRange"][unit.type] / 8
        return sight_range

    def unit_max_shield(self, unit_id, ally):
        # These are the biggest I've seen, there is no info about this
        if ally:
            unit = self.agents[unit_id]
        else:
            unit = self.enemies[unit_id]

        return tcc.staticvalues["maxShields"][unit.type]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """

        unit = self.get_unit_by_id(agent_id)

        nf_al = 4
        nf_en = 4

        move_feats = np.zeros(self.n_actions_no_attack - 2, dtype=np.float32)  # exclude no-op & stop
        enemy_feats = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.x
            y = unit.y
            sight_range = self.unit_sight_range(agent_id)

            avail_actions = self.get_avail_agent_actions(agent_id)

            for m in range(self.n_actions_no_attack - 2):
                move_feats[m] = avail_actions[m + 2]

            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.x
                e_y = e_unit.y
                dist = self.distance(x, y, e_x, e_y)

                if dist < sight_range and e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y

            # Place the features of the agent himself always at the first place
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.x
                al_y = al_unit.y
                dist = self.distance(x, y, al_x, al_y)

                if dist < sight_range and al_unit.health > 0:  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

        agent_obs = np.concatenate((move_feats.flatten(),
                                    enemy_feats.flatten(),
                                    ally_feats.flatten()))

        agent_obs = agent_obs.astype(dtype=np.float32)

        if self.debug_inputs:
            print("***************************************")
            print("Agent: ", agent_id)
            print("Available Actions\n", self.get_avail_agent_actions(agent_id))
            print("Move feats\n", move_feats)
            print("Enemy feats\n", enemy_feats)
            print("Ally feats\n", ally_feats)
            print("***************************************")

        return agent_obs

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        nf_al = 4
        nf_en = 3

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))
        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.x
                y = al_unit.y

                ally_state[al_id, 0] = al_unit.health / al_unit.max_health  # health
                ally_state[al_id, 1] = al_unit.groundCD / al_unit.maxCD  # cool-down for ground weapon
                ally_state[al_id, 2] = (x - center_x) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (y - center_y) / self.max_distance_y  # relative Y

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.x
                y = e_unit.y

                enemy_state[e_id, 0] = e_unit.health / e_unit.max_health  # health
                enemy_state[e_id, 1] = (x - center_x) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (y - center_y) / self.max_distance_y  # relative Y

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        state = state.astype(dtype=np.float32)

        if self.debug_inputs:
            print("------------ STATE ---------------")
            print("Ally state\n", ally_state)
            print("Enemy state\n", enemy_state)
            print("Last action\n", self.last_action)
            print("----------------------------------")

        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        nf_al = 4
        nf_en = 3

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions

        return size

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # Cannot do no-op as alive
            avail_actions = [0] * self.n_actions

            # Stopping should be allowed
            avail_actions[1] = 1

            if unit.y + self._move_amount < self.map_play_area_max[1]:
                avail_actions[2] = 1
            if unit.y - self._move_amount > self.map_play_area_min[1]:
                avail_actions[3] = 1
            if unit.x + self._move_amount < self.map_play_area_max[0]:
                avail_actions[4] = 1
            if unit.x - self._move_amount > self.map_play_area_min[0]:
                avail_actions[5] = 1

            shooting_range = self.unit_shoot_range(agent_id)

            # Unit can only attack enemies that are alive and in shooting range
            for e_id, e_unit in self.enemies.items():
                if e_unit.health > 0:
                    dist = self.distance(unit.x, unit.y, e_unit.x, e_unit.y)
                    if dist <= shooting_range:
                        avail_actions[e_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # Only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):

        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_obs_size(self):
        """ Returns the shape of the observation """
        nf_al = 4
        nf_en = 4

        move_feats = self.n_actions_no_attack - 2
        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al

        return move_feats + enemy_feats + ally_feats

    def close(self):
        print("Closing StarCraftI")
        self.controller.close()

    def render(self):
        pass

    def save_units(self):
        # Called after initialising the map to remember the locations of units
        self.agents_orig = {}
        self.enemies_orig = {}

        self._obs = self.controller.recv()

        for unit in self._obs.units[0]:
            self.agents_orig[len(self.agents_orig)] = unit
        for unit in self._obs.units[1]:
            self.enemies_orig[len(self.enemies_orig)] = unit

        assert len(self.agents_orig) == self.n_agents, "Incorrect number of agents: " + str(len(self.agents_orig))
        assert len(self.enemies_orig) == self.n_enemies, "Incorrect number of enemies: " + str(len(self.enemies_orig))

    def restore_units(self):
        # Restores the original state of the game
        pass

    def kill_all_units(self):
        self.agents = {}
        self.enemies = {}

        while len(self._obs.units[0]) > 0 or len(self._obs.units[1]) > 0:
            commands = [[tcc.set_combine_frames, 1]]
            for j in range(2):
                for i in range(len(self._obs.units[j])):
                    u = self._obs.units[j][i]
                    command = [
                        tcc.command_openbw,
                        tcc.openbwcommandtypes.KillUnit,
                        u.id,
                    ]
                    commands.append(command)

            self.controller.send(commands)
            self._obs = self.controller.recv()

        self.controller.send([
            [tcc.set_combine_frames, self._step_mul],
        ])

        self._obs = self.controller.recv()

    def init_units(self):

        counter = 1
        while True:

            self.agents = {}
            self.enemies = {}

            ally_units = [deepcopy(unit) for unit in self._obs.units[0]]
            ally_units_sorted = sorted(ally_units, key=attrgetter('type', 'x', 'y'), reverse=False)

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug_inputs:
                    print("Unit %d is %d, x = %.1f, y = %1.f" % (
                        len(self.agents), self.agents[i].type, self.agents[i].x, self.agents[i].y))

            for unit in self._obs.units[1]:
                # for unit in self._obs.observation.raw_data.units:
                # if unit.owner == 2: # agent
                self.enemies[len(self.enemies)] = deepcopy(unit)

            if self.agents == {}:
                counter += 1

            if len(self.agents) == self.n_agents and len(self.enemies) == self.n_enemies:
                # print('Successfully spawned all {} agents after iteration {}'.format(self.n_agents, counter))
                for e_id, e_unit in self.enemies.items():
                    if self._episode_count == 1:
                        self.max_reward += e_unit.max_health  # + unit.max_shield
                return

            # Send actions
            self.controller.send([])

            self._obs = self.controller.recv()
            # self.full_restart()  # TODO: Implement condition to trigger full restart in case the client has crashed

    def update_units(self):
        # This function assumes that self._obs is up-to-date
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_agent_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.units[0]:
                if al_unit.id == unit.id:
                    self.agents[al_id] = deepcopy(unit)
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # means agent unit is dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.units[1]:
                if e_unit.id == unit.id:
                    self.enemies[e_id] = deepcopy(unit)
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # means agent unit is dead
                e_unit.health = 0

        if n_ally_alive == 0 and n_enemy_alive > 0:
            return -1  # loss
        if n_ally_alive > 0 and n_enemy_alive == 0:
            return 1  # win
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0  # tie, not sure if this is possible

        return None

    def get_unit_by_id(self, a_id):
        return self.agents[a_id]

    def get_stats(self):
        stats = {}
        stats["battles_won"] = self.battles_won
        stats["battles_game"] = self.battles_game
        stats["win_rate"] = self.battles_won / self.battles_game
        stats["timeouts"] = self.timeouts
        stats["restarts"] = self.force_restarts

        # print("Stats: {}\n".format(stats))
        return stats

    def _add_deepcopy_support(self):
        tc.replayer.Order.__deepcopy__ = _deepcopy_order
        tc.replayer.UnitCommand.__deepcopy__ = _deepcopy_unit_command
        tc.replayer.Unit.__deepcopy__ = _deepcopy_unit


def _deepcopy_order(self, memo):
    o = tc.replayer.Order()
    o.first_frame = self.first_frame
    o.type = self.type
    o.targetId = self.targetId
    o.targetX = self.targetX
    o.targetY = self.targetY
    return o


def _deepcopy_unit_command(self, memo):
    c = tc.replayer.UnitCommand()
    c.frame = self.frame
    c.type = self.type
    c.targetId = self.targetId
    c.targetX = self.targetX
    c.targetY = self.targetY
    c.extra = self.extra
    return c


def _deepcopy_unit(self, memo):
    u = tc.replayer.Unit()
    u.id = self.id
    u.x = self.x
    u.y = self.y
    u.health = self.health
    u.max_health = self.max_health
    u.shield = self.shield
    u.max_shield = self.max_shield
    u.energy = self.energy
    u.maxCD = self.maxCD
    u.groundCD = self.groundCD
    u.airCD = self.airCD
    u.flags = self.flags
    u.visible = self.visible
    u.type = self.type
    u.armor = self.armor
    u.shieldArmor = self.shieldArmor
    u.size = self.size
    u.pixel_x = self.pixel_x
    u.pixel_y = self.pixel_y
    u.pixel_size_x = self.pixel_size_x
    u.pixel_size_y = self.pixel_size_y
    u.groundATK = self.groundATK
    u.airATK = self.airATK
    u.groundDmgType = self.groundDmgType
    u.airDmgType = self.airDmgType
    u.groundRange = self.groundRange
    u.airRange = self.airRange

    u.orders = deepcopy(self.orders)
    u.command = deepcopy(self.command)

    u.velocityX = self.velocityX
    u.velocityY = self.velocityY

    u.playerId = self.playerId
    u.resources = self.resources

    return u


class StatsAggregator():

    def __init__(self):
        self.last_stats = None
        self.stats = []
        pass

    def aggregate(self, stats, add_stat_fn):

        current_stats = {}
        for stat in stats:
            for _k, _v in stat.items():
                if not (_k in current_stats):
                    current_stats[_k] = []
                if _k in ["win_rate"]:
                    continue
                current_stats[_k].append(_v)

        # average over stats
        aggregate_stats = {}
        for _k, _v in current_stats.items():
            if _k in ["win_rate"]:
                aggregate_stats[_k] = np.mean(
                    [(_a - _b) / (_c - _d) for _a, _b, _c, _d in zip(current_stats["battles_won"],
                                                                     [0] * len(current_stats[
                                                                                   "battles_won"]) if self.last_stats is None else
                                                                     self.last_stats["battles_won"],
                                                                     current_stats["battles_game"],
                                                                     [0] * len(current_stats[
                                                                                   "battles_game"]) if self.last_stats is None else
                                                                     self.last_stats["battles_game"])
                     if (_c - _d) != 0.0])
            else:
                aggregate_stats[_k] = np.mean(
                    [_a - _b for _a, _b in zip(_v, [0] * len(_v) if self.last_stats is None else self.last_stats[_k])])

        # add stats that have just been produced to tensorboard / sacred
        for _k, _v in aggregate_stats.items():
            add_stat_fn(_k, _v)

        # collect stats for logging horizon
        self.stats.append(aggregate_stats)
        # update last stats
        self.last_stats = current_stats
        pass

    def log(self, log_directly=False):
        assert not log_directly, "log_directly not supported."
        logging_str = " Win rate: {}".format(_seq_mean([stat["win_rate"] for stat in self.stats])) \
                      + " Timeouts: {}".format(_seq_mean([stat["timeouts"] for stat in self.stats])) \
                      + " Restarts: {}".format(_seq_mean([stat["restarts"] for stat in self.stats]))

        # flush stats
        self.stats = []
        return logging_str
