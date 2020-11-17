# Instructions: Aliengo Standing-up Task using Reinforcement Learning, Integration of ROS-Gazebo, Openai and Tensoflow.


* Openai-Ros-Gazebo is implemented from 
https://bitbucket.org/theconstructcore/openai_ros.git 
* PPO RL trainig is implemented from 
https://github.com/nav74neet/ppo_gazebo_tf
* DQN and DDQN is implemented from
https://github.com/pythonlessons/Reinforcement_Learning/blob/master/02_CartPole-reinforcement-learning_DDQN/Cartpole_DDQN_TF2.py
* Aliengo Gazebo is implemented from 
https://github.com/unitreerobotics/aliengo_ros and https://github.com/unitreerobotics/laikago_ros


## Dependencies:
Requirment:
* Ubuntu 16.04 or 18.04
* gym
*  python 3.6 or above
*  tensoflow 1.4 or above
```
sudo apt install python3-pip
pip3 install gym
pip3 install --upgrade  tensorflow==1.4
pip3 install rospkg
pip3 install defusedxml
sudo apt-get install python3-git
```
ROS kinetic  or melodic
Gazebo8 or above
```
ros-melodic-gazebo8-ros 
ros-melodic-gazebo8-ros-control
ros-melodic-gazebo8-ros-pkgs
ros-melodic-gazebo8-ros-dev
```
ROS related packages for simulation
```
sudo apt-get install ros-melodic-controller-manager ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-joint-state-controller ros-melodic-effort-controllers ros-melodic-velocity-controllers ros-melodic-position-controllers ros-melodic-robot-controllers ros-melodic-robot-state-publisher
```


## Building ROS workspace:

make a workspace folder (for example: learning_ws) and copy src file into here.
* `cd ~/learning_ws`
* `catkin_make`
to make sure all ros related packages are installed, go to learning_ws
* `source devel/setup.bash`
* `rosdep install openai_ros`<br>
* `rosdep install learning_ros`<br>
* `rosdep install aliengo_gazebo`<br>
* `rosdep install laikago_controller`<br>

## RL Related Configuration
* Go to src/learning_ros/config and open aliengo_stand.yaml
* change ros_ws_abspath with your worksapce path
/home/asoro/current_hw/deep_learning/learning_ws --> /home/your_ros_workspace
you can aslo change goal paramaters such as desired goal X, Z and pitch angle. 
* Go to main branch and make executable all .sh files
* Go to aliengo/aliengo_gazebo/script and make executable sim_odometry_converter.py
* Go to learning_ros/script and make executable all python files


### openai_ros
Ros-gazebo communication to open-ai gym environment. 
Adapted for aliengo standing up mission. 
robot_envs/aliengo_env.py is robot environment (communicates between ROS related joint controller and ROS related topics), task_envs/aliengo_task.py is task enviorment. We defined our observations, reward and actions to make standing-up.
### learning_ros
includes python code of PPO training,testing, DDQN training and testing. Their respective models can be found accoring to their naming. 

### aliengo_description:
including mesh, urdf and xacro files of quadrupedal robot named Aliengo A1
### aliengo_gazebo:
Spawns aliengo robot in Gazebo enviornment(aliengo_empty_world.launch) Note that if gui param is false, you need to turn it true for simulation rendering. 
### laikago_controller:
Laikago's default joint controller, subscribes joint commands via topics
### laikago_msgs:
Laikago's default msgs


## How to Train
Open first terminal;
* `roscore`
Open second terminal;
go to /learning_ws and 
To train PPO agent in gazebo 
* `source devel/setup.bash`
then go to /learning_ws/src/learning_ros/script and 
* `python3 ppo_train_aliengo.py`
OR 
go to /learning_ws and 
* `./run_ppo_train.sh`

Similarly to train with DDQN (double deep Q network) or DQN (you can change by  ddqn = true to false)
* `./run_ddqn_train.sh`
* `./run_dqn_train.sh`
## How to Test
Open first terminal;
* `roscore`
Open second terminal;
go to /learning_ws and 
To test learned PPO model
* `./run_ppo_test.sh`
To test learned DDQN or DQN model
* `./run_ddqn_test.sh`
* `./run_dqn_test.sh`

* Note-1: If you want to see in Gazebo GUI go to aliengo/aliengo_gazebo/launch , open aliengo_empty_world.launch
change <arg name="gui" default="false"/> to true (we  make this false so that our computer can handle learning process in realtime)
* Note-2: After closing learining agent sometimes gazebo doesnt close properly or Gazebo and ROS related packages might crashed. To resolve this issue downlaod "htop" and SIGKILL gazebo. 

## Results
Gazebo and Rviz visulization of Desired Stand-up Behaviour
![Alt text](docs/gazebo_rviz_result.png?raw=true "Snapshots of Standing-up Position: (Left) Gazebo Physic Simulator (Right) Rviz Sensor-Reward Visualization where green arrow indicate robot reached the goal pose , red cube indicate 1st reward, blue cube 2nd rewardand green cube 3th reward")
### Training
Double DQN and PPO-Clip is trained around 260 episodes. Learning terminates after 10 consecutive high reward.

![Training Reward-Episode Graphs: (Left) PPO-Clip Agent, (Right)DDQN Agent](docs/train_graphs.png?raw=true "Training Reward-Episode Graphs: (Left) PPO-Clip Agent, (Right)DDQN Agent")


![PPO-Clip Training](docs/train_ppo.gif?raw=true "PPO-Clip Training")
### Testing
Double DQN and PPO-Clip is tested in 100 Episodes. PPO-CLip succes rate 94/100, Double DQN score 88/100
Testing Reward-Episode Graphs: (Left) PPO-Clip Agent, (Right)DDQN Agent
![Alt text](docs/testing_graphs.png?raw=true "Testing Reward-Episode Graphs: (Left) PPO-Clip Agent, (Right)DDQN Agent")

Robot is able to stand up and stay in stand up position
PPO-Clip Testing
![Alt text](docs/testing_ppo.gif?raw=true "PPO-Clip Testing")

Robot is able to stand up and  but directly reaches terminal state instead of staying in stand-up position 
Double-DQN Testing
![Screenshot](docs/testing_ddqn.gif?raw=true "Double-DQN Testing")






