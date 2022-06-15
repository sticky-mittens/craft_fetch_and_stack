import numpy as np
import sys
import copy

class option_0():
    def __init__(self):
        self.thresh = 0.05
        self.block_name = "some_block"
        self.height = 0.6
        self.grabbed = False
        self.counter = 0

    def initial_predicate(self, obs):
        ''' Checks if the nearest block is not in grip or goal location and can be grabbed '''

        # Find the current goals for each block
        object_goals = find_goal_states(obs)

        self.block_name = find_closer_object_name(obs)

        cl_object = obs["annotated_obs"][self.block_name]

        # Check if the block is in destination
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]

        goal_dist = find_dist(rel_pos)

        if (np.max(np.abs(cl_object['rel_pos'])) > self.thresh) and \
        (goal_dist > self.thresh):
            # print("Option 0 triggered  for object - ", self.block_name)
            self.grabbed = False
            return True

        return False

    def compute_option_policy(self, obs):
        ''' Computes the low level action  '''


        closer_obj = obs["annotated_obs"][self.block_name]

        # x_y_dist = np.linalg.norm(closer_obj['rel_pos'][0:2])
        dist = find_dist(closer_obj['rel_pos'])
        torques = 0


        if self.grabbed:
            gripper = -1
            self.counter = 0
            torques = np.array([0, 0, 50 * (self.height - obs["annotated_obs"]["grip_pos"][2])])
        elif (dist > (self.thresh - 0.02)):
            gripper = 1
            torques = feedback_reach_object(closer_obj['rel_pos'])
        elif dist < (self.thresh - 0.02) :
            gripper = -1
            torques = feedback_reach_object(closer_obj['rel_pos'])
            self.counter += 1
            if self.counter > 5 :
                self.grabbed = True

        action = np.concatenate([torques, np.array([gripper])])

        return action

    def termination_condition(self, obs):
        # Check if the block location and grip position are the same, and this thing is in the air.

        cl_object = find_closer_object(obs)

        dist = find_dist(cl_object['rel_pos'])


        if (dist < (self.thresh - 0.02) ) and (obs["annotated_obs"]["grip_pos"][2] > self.height) and (self.grabbed):
            # print("When exiting option 0, distance - ", dist)
            self.counter = 0
            self.grabbed = False
            return True

        return False


class option_1():
    def __init__(self):
        self.thresh = 0.05
        self.height = 0.7
        self.block_name = "some_block"
        self.placed_counter = 0
        self.took_off = False

    def initial_predicate(self, obs):
        ''' Checks if there is a block in grip, moves it to the goal '''

        cl_object = find_closer_object(obs)

        self.block_name = find_closer_object_name(obs)

        # Find the block in hand

        if (find_dist(cl_object['rel_pos']) < ( self.thresh - 0.01 )) and \
        ( (obs["annotated_obs"]["grip_pos"][2])  > 0):
            # print("Option 1 is true for object - ", self.block_name)
            return True

        return False

    def compute_option_policy(self, obs):
        ''' Computes the low level action  '''

        if self.block_name == "some_block" :
            assert(self.initial_predicate(obs))

        # Find the current goals for each block
        object_goals = find_goal_states(obs)

        # # For the current block find the goal
        # object_name = find_closer_object_name(obs)

        # Check if the block is in destination
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]

        goal_dist = find_dist(rel_pos)


        if goal_dist > self.thresh :
            gripper = -1
            torques = feedback_reach_object(rel_pos)
        elif self.placed_counter < 10:
            gripper = -1
            torques = feedback_reach_object(rel_pos)
            self.placed_counter += 1
        elif self.placed_counter < 20 :
            self.placed_counter += 1
            gripper = 1
            torques = np.array([0, 0, 70 * (self.height - obs["annotated_obs"]["grip_pos"][2])])
        else:
            gripper = 1
            torques = np.array([0, 0, 0])
            self.took_off = True

        # print(" placed_counter = ", self.placed_counter, " took off flag - ", self.took_off)

        action = np.concatenate([torques, np.array([gripper])])

        return action


    def termination_condition(self, obs):
        ''' Check if the block in grip is at it's goal '''

        # Find the current goals for each block
        object_goals = find_goal_states(obs)

        # For the current block find the goal
        object_name = find_closer_object_name(obs)

        # Check if the block is in destination
        rel_pos = object_goals[object_name] - obs["annotated_obs"][object_name]["pos"]
        dist_from_target = np.linalg.norm(rel_pos)


        # if (dist_from_target < (self.thresh-0.02)):
        if self.took_off and self.placed_counter >= 10:
            self.placed_counter = 0
            self.took_off = False
            return True
        else :
            return False


class place_object():
    def __init__(self, block_name):
        self.thresh = 0.05
        self.height = 0.65
        self.block_name = block_name
        self.grabbed = False
        self.picked_counter = 0
        self.placed_counter = 0
        self.took_off = False
        self.picking_up_part = False



    def initial_predicate(self, obs):
        ''' Checks if there is a block in grip, moves it to the goal '''

        object_of_interest = obs["annotated_obs"][self.block_name]

        # Find the current goals for each block
        object_goals = find_goal_states(obs)

        # Check if the block is in destination
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]

        goal_dist = find_dist(rel_pos)

        self.reach_func = reach_object(self.block_name)
        self.pick_and_reach = pick_and_reach_goal(self.block_name)
        self.release_and_lift = release_and_lift(self.block_name)
        self.state = -1

        if (np.max(np.abs(object_of_interest['rel_pos'])) > self.thresh) and \
        (goal_dist > (self.thresh-0.02)):
            self.grabbed = False
            self.picked_counter = 0
            self.placed_counter = 0
            self.pickup_counter = 0
            self.took_off = False
            self.picking_up_part = False



            return True

        return False

    def compute_option_policy(self, obs):
        ''' Computes the low level action  '''



        object_of_interest = obs["annotated_obs"][self.block_name]

        grabbing_dist = find_dist(object_of_interest['rel_pos'])

        # Find the current goals for each block
        object_goals = find_goal_states(obs)

        # Check if the block is in destination
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]

        goal_dist = find_dist(rel_pos)

        if (grabbing_dist < (self.thresh - 0.02) ) and \
        (obs["annotated_obs"]["grip_pos"][2] > self.height) and (self.grabbed):
            # print("When exiting option 0, distance - ", dist)
            self.picking_up_part = True
            self.picked_counter = 0
            self.grabbed = False

        torques = 0
        if not self.picking_up_part :
            if self.grabbed:
                gripper = -1
                self.picked_counter = 0
                torques = np.array([0, 0, 100 * (self.height - obs["annotated_obs"]["grip_pos"][2])])
            elif (grabbing_dist > (self.thresh - 0.03)):
                gripper = 1
                torques = feedback_reach_object(object_of_interest['rel_pos'])
            elif grabbing_dist < (self.thresh - 0.03) :
                gripper = -1
                torques = feedback_reach_object(object_of_interest['rel_pos'])
                self.picked_counter += 1
                if self.picked_counter > 5 :
                    self.grabbed = True
        else:
            if goal_dist > (self.thresh - 0.03):
                gripper = -1
                torques = feedback_reach_object(rel_pos)
            elif self.placed_counter < 8:
                gripper = -1
                torques = feedback_reach_object(rel_pos)
                self.placed_counter += 1
            elif self.placed_counter < 15:
                self.placed_counter += 1
                gripper = 1
                torques = np.array([0, 0, 100 * (self.height - obs["annotated_obs"]["grip_pos"][2])])
            else:
                gripper = 1
                torques = np.array([0, 0, 0])
                self.took_off = True

        action = np.concatenate([torques, np.array([gripper])])

        return action

    def compute_option_policy_2(self, obs):
        ''' Computes the low level action  '''

        action = np.array([0, 0, 0, 0])


        if self.state == -1 and self.reach_func.predicate(obs) :
            self.state = 0

        if self.state == 0 and self.reach_func.termination(obs):
            _ =  self.pick_and_reach.predicate(obs)
            self.state = 1
            self.timer = 0

        if self.state == 1 and  self.pick_and_reach.termination(obs) and self.timer > 10:
            _ =  self.release_and_lift.predicate(obs)
            self.state = 2



        if self.state == 0 :
            action = self.reach_func.compute_option_policy(obs)
        elif self.state == 1:
            self.timer += 1
            action = self.pick_and_reach.compute_option_policy_2(obs)
        elif self.state == 2 :
            action = self.release_and_lift.compute_option_policy(obs)
            self.timer = 0

        if self.state == 2 and  self.release_and_lift.termination(obs):
            self.state = -1
            self.took_off = True
            self.placed_counter = 13 # 1 more than 12

        return action


    def termination_condition(self, obs):
        ''' Check if the block in grip is at it's goal '''

        # Find the current goals for each block
        object_goals = find_goal_states(obs)

        # # For the current block find the goal
        # object_name = find_closer_object_name(obs)

        # Check if the block is in destination
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]
        dist_from_target = np.linalg.norm(rel_pos)


        # if (dist_from_target < (self.thresh-0.02)):
        if self.took_off and self.placed_counter >= 12:
            self.pickup_counter = 0
            # self.grabbed = False
            # self.picked_counter = 0
            # self.placed_counter = 0
            # self.took_off = False
            # self.picking_up_part = False

            return True
        else :
            return False

    def accomplished(self, obs):
        object_goals = find_goal_states(obs)
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]
        dist_from_target = np.linalg.norm(rel_pos)
        goal_height = object_goals[self.block_name][2]

        # if dist_from_target < (self.thresh + 0.02) and (obs["annotated_obs"]['grip_pos'][2] > (self.height - 0.05)):
        if dist_from_target < (self.thresh - 0.02) and (obs["annotated_obs"]['grip_pos'][2] > (goal_height + 0.05)):
            # print("Checking accomplishment for block - ", self.block_name, " True")

            # self.grabbed = False
            # self.picked_counter = 0
            # self.placed_counter = 0
            # self.took_off = False
            # self.picking_up_part = False
            # print("Terminated by goal accomplished")
            return True
        else:
            # print("Checking accomplishment for block - ", self.block_name, " False")
            return False


class do_nothing():
    def __init__(self):
        self.thresh = 0.05

    def initial_predicate(self, obs):
        return True


    def compute_option_policy(self, obs):
        torques = np.array([0, 0, 0, 0])

        return torques

    def termination_condition(self, obs):
        return True

class reach_object():
    '''
    Reach and leave gripper open
    '''

    def __init__(self, block_name):
        self.thresh = 0.05
        self.block_name = block_name


    def predicate(self, obs):
        ''' basically gripper is empty and block is not in goal'''


        rel_pos = obs["annotated_obs"][self.block_name]["rel_pos"]
        obj_dist = find_dist(rel_pos)

        if obj_dist > (self.thresh):
            # Internal states
            self.gripper_open = False
            return True

        return False

    def compute_option_policy(self, obs):
        ''' computes the low level policy '''

        object_of_interest = obs["annotated_obs"][self.block_name]
        gripper = 1
        torques = feedback_reach_object(object_of_interest['rel_pos'])
        action = np.concatenate([torques, np.array([gripper])])

        return action


    def termination(self, obs):
        ''' The target object is in gripper, and gripper is open '''


        rel_pos = obs["annotated_obs"][self.block_name]["rel_pos"]
        dist_from_target = np.linalg.norm(rel_pos)

        if dist_from_target < (self.thresh - 0.02):
            # print("---------------Termination of reach for block ", self.block_name, " is true")
            return True

        return False

class pick_and_reach_goal():
    '''
    Close gripper and you reach the goal
    '''

    def __init__(self, block_name):
        self.thresh = 0.05
        self.block_name = block_name
        self.height = 0.65
        self.prev_torque = np.array([0, 0, 0])


    def predicate(self, obs):
        ''' Gripper position is good '''

        object_of_interest = obs["annotated_obs"][self.block_name]

        # Check if the block is in destination
        rel_pos = object_of_interest['rel_pos']
        dist_from_target = np.linalg.norm(rel_pos)

        if dist_from_target < self.thresh :
            # Internal states
            self.grabbed_counter = 0
            self.pickup_counter = 0
            self.reach_goal_ctr = 0

            return True

        return False

    def compute_option_policy_1(self, obs):
        ''' Close the gripper, reach goal '''

        if self.grabbed_counter < 10 :
            self.grabbed_counter += 1
            gripper = -1
            torques = np.array([0, 0, 0])
            action = np.concatenate([torques, np.array([gripper])])
            return action
        elif self.pickup_counter < 12:
            self.pickup_counter += 1
            gripper = -1
            # torques = np.array([0, 0, 100 * (self.height - obs["annotated_obs"]["grip_pos"][2])])
            torques = np.array([0, 0, 0.6])
            action = np.concatenate([torques, np.array([gripper])])
            return action
        elif self.pickup_counter == 12:
            # self.reach_goal_ctr += 1
            # Check if the block is in destination
            object_goals = find_goal_states(obs)
            rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]
            gripper = -1
            torques = feedback_reach_object_v2(rel_pos)
            action = np.concatenate([torques, np.array([gripper])])
            return action


        gripper = -1
        torques = np.array([0, 0, 0])
        action = np.concatenate([torques, np.array([gripper])])
        return action

    def compute_option_policy_2(self, obs):
        ''' Close the gripper, reach goal '''

        if self.grabbed_counter < 5 :
            self.grabbed_counter += 1
            gripper = -1
            torques = np.array([0, 0, 0])
            action = np.concatenate([torques, np.array([gripper])])
            return action
        elif self.pickup_counter < 17:

            self.pickup_counter += 1
            gripper = -1

            object_goals = copy.deepcopy(find_goal_states(obs))
            # Reach slightly above the target goal location
            object_goals[self.block_name][2] += 0.1
            rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]
            torques = feedback_reach_object_v4(rel_pos)
            action = np.concatenate([torques, np.array([gripper])])
            # torques = np.array([0, 0, 100 * (self.height - obs["annotated_obs"]["grip_pos"][2])])
            # torques = np.array([0, 0, 0.6])
            # action = np.concatenate([torques, np.array([gripper])])
            return action
        elif self.pickup_counter == 17:
            # self.reach_goal_ctr += 1
            # Check if the block is in destination
            object_goals = find_goal_states(obs)
            rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]
            gripper = -1
            torques = feedback_reach_object_v3(rel_pos)

            action = np.concatenate([torques, np.array([gripper])])
            return action


        gripper = -1
        torques = np.array([0, 0, 0])
        action = np.concatenate([torques, np.array([gripper])])
        return action

    def termination(self, obs):
        ''' Desired block is in goal '''
        object_goals = find_goal_states(obs)
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]
        goal_height = object_goals[self.block_name][2]
        current_height = obs["annotated_obs"]["grip_pos"][2]

        if np.linalg.norm(rel_pos) < (self.thresh-0.02) :
            # print("---------------Termination of pick n reach for block ", self.block_name, " is true rel pos - ", np.linalg.norm(rel_pos) )

            self.grabbed_counter = 0
            self.pickup_counter = 0
            self.reach_goal_ctr = 0
            return True

        return False

class release_and_lift():
    '''
    Release the gripper and lift hand
    '''

    def __init__(self, block_name):
        self.height = 0.65
        self.thresh = 0.05

        self.block_name = block_name

    def predicate(self, obs = None):
        current_height = obs["annotated_obs"]["grip_pos"][2]

        object_goals = find_goal_states(obs)
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]
        self.gripper_open_counter = 0
        self.lift_count = 0
        if current_height < (self.height) and (np.linalg.norm(rel_pos) < self.thresh):
            # print("---------------Termination of release for block  is true")
            self.gripper_open_counter = 0
            self.lift_count = 0
            return True
        else :
            return False

    def compute_option_policy(self, obs):
        ''' Open the gripper completely and lift hand without changing x-y coord  '''

        if self.gripper_open_counter < 5 :
            self.gripper_open_counter += 1
            gripper = 1
            torques = np.array([0, 0, 0])
            action = np.concatenate([torques, np.array([gripper])])
            return action
        elif self.lift_count < 8:
            self.lift_count += 1
            self.grabbed_counter = 0
            gripper = 1
            torques = np.array([0, 0, 100 * (self.height - obs["annotated_obs"]["grip_pos"][2])])
            # torques = np.array([0, 0, 1])
            action = np.concatenate([torques, np.array([gripper])])
            return action

        gripper = -1
        torques = np.array([0, 0, 0])
        action = np.concatenate([torques, np.array([gripper])])
        return action


    def termination(self, obs):
        ''' Check if the height is reached '''

        current_height = obs["annotated_obs"]["grip_pos"][2]
        object_goals = find_goal_states(obs)
        goal_height = object_goals[self.block_name][2]
        rel_pos = object_goals[self.block_name] - obs["annotated_obs"][self.block_name]["pos"]

        # if current_height > (self.height - 0.05) and (np.linalg.norm(rel_pos) < (self.thresh + 0.01)):
        if current_height > (goal_height + 0.05) and (np.linalg.norm(rel_pos) < (self.thresh - 0.02)):
            # print("---------------Termination of release for block  is true")

            self.gripper_open_counter = 0
            self.lift_count = 0
            return True
        else :
            return False


def feedback_reach_object(rel_pos):
    torque = np.array([0, 0 , 0])

    thresh = 0.02

    if np.abs(rel_pos[0]) > thresh or np.abs(rel_pos[1]) > thresh:
        for i in range(len(rel_pos)):
            if np.abs(rel_pos[i]) > thresh and i != 2:
                torque[i] = 100 * rel_pos[i]
        torque = np.array(torque)
        return torque

    else:
        torque[2] = 100 * rel_pos[2]
        torque = np.array(torque)
        return torque

    for i in range(len(rel_pos)):
        if np.abs(rel_pos[i]) > thresh:
            torque[i] = 70 * rel_pos[i]
            break

    torque = np.array(torque)
    return torque

def feedback_reach_object_v2(rel_pos):
    torque = np.array([0, 0 , 0])

    thresh = 0.02

    if np.abs(rel_pos[0]) > thresh or np.abs(rel_pos[1]) > thresh:
        for i in range(len(rel_pos)):
            if np.abs(rel_pos[i]) > thresh and i != 2:
                torque[i] = 100 * rel_pos[i]
        torque = np.array(torque)
        return torque
    else:
        if np.abs(rel_pos[2]) > 0.2 :
            torque[2] = 100 * rel_pos[2]
        else:
            torque[2] = 80 * rel_pos[2]
        torque = np.array(torque)
        return torque

    for i in range(len(rel_pos)):
        if np.abs(rel_pos[i]) > thresh:
            torque[i] = 100 * rel_pos[i]
            break

    torque = np.array(torque)
    return torque

def feedback_reach_object_v3(rel_pos):
    torque = np.array([0, 0 , 0])

    thresh = 0.02

    if (np.abs(rel_pos[2])) > thresh:
        torque[2] = 100 * rel_pos[2]
        torque = np.array(torque)
        torque = np.clip(torque, -0.5, 0.5)

        return torque
    else:
        for i in range(len(rel_pos)):
            if np.abs(rel_pos[i]) > thresh and i != 2:
                torque[i] = 100 * rel_pos[i]
        torque = np.array(torque)
        torque = np.clip(torque, -0.9, 0.9)
        return torque

    for i in range(len(rel_pos)):
        if np.abs(rel_pos[i]) > thresh:
            torque[i] = 100 * rel_pos[i]
            break

    torque = np.array(torque)
    return torque


def feedback_reach_object_v4(rel_pos):
    torque = np.array([0, 0 , 0])

    thresh = 0.1

    if np.abs(rel_pos[2]) > thresh :
        torque[2] = 100 * rel_pos[2]
    else:
        for i in range(len(rel_pos)):
            torque[i] = 100 * rel_pos[i]
        torque = np.clip(torque, -0.6, 0.6)

    # torque = np.array(torque)
    return torque

def predicate(rel_pos):

    thresh = 0.05

    for idx in range(len(rel_pos)):
        if np.abs(rel_pos[idx]) > thresh :
            return False

    return True

def implement_option(object_name, obs):

    rel_pos = obs["annotated_obs"][object_name]["rel_pos"]
    gripper = 0

    if predicate(rel_pos):
        gripper = -1
    else:
        gripper = 1

    torques = feedback_reach_object(rel_pos)

    action = np.concatenate([torques, np.array([gripper])])

    return action


def find_goal_states(obs):
    goal_env = obs['desired_goal']
    pos_size = 3
    no_of_objs = int(len(goal_env) / pos_size - 1)

    goals = {}
    for obj_id in range(no_of_objs):
        goals["object_" + str(obj_id)] = goal_env[obj_id * pos_size : (obj_id + 1) * pos_size]

    return goals

def find_dist(arr):
    return np.max(np.abs(arr))

def find_closer_object(obs):
    annotated_obs = obs["annotated_obs"]
    closest_object = {}
    dist = np.inf
    for key in annotated_obs.keys():
        if (key.find("object") > -1):
            if(dist > find_dist(annotated_obs[key]['rel_pos']) ) :
                dist = find_dist(annotated_obs[key]['rel_pos'])
                closest_object = annotated_obs[key]


    return closest_object

def find_object_list(obs):
    annotated_obs = obs["annotated_obs"]
    closest_object = {}
    obj_list = []
    dist = np.inf
    for key in annotated_obs.keys():
        if (key.find("object") > -1):
            obj_list.append(key)

    return obj_list

def find_closer_object_name(obs):
    annotated_obs = obs["annotated_obs"]
    closest_object = "k"
    dist = np.inf
    for key in annotated_obs.keys():
        if (key.find("object") > -1):
            if(dist > find_dist(annotated_obs[key]['rel_pos']) ) :
                dist = find_dist(annotated_obs[key]['rel_pos'])
                closest_object = key


    return closest_object


def find_closer_object_dist(obs):

    closer_object = find_closer_object(obs)
    dist = find_dist(closer_object['rel_pos'])

    return dist
