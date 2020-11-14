# Instructions: Aliengo Standing-up Task using Reinforcement Learning, Integration of ROS-Gazebo, Openai and Tensoflow.
 

https://bitbucket.org/theconstructcore/openai_ros.git
https://github.com/pythonlessons/Reinforcement_Learning/blob/master/02_CartPole-reinforcement-learning_DDQN/Cartpole_DDQN_TF2.py


## Dependencies:
Requirment:
Ubuntu 16.04 or 18.04
gym
python 3.6 or above
tensoflow 1.4 or above
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
ros-melodic-gazebo8-ros ros-melodic-gazebo8-ros-control ros-melodic-gazebo8-ros-pkgs ros-melodic-gazebo8-ros-dev
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

To train PPO agent in gazebo go to /learning_ws
* `source devel/setup.bash`
then go to /learning_ws/src/learning_ros/script and 
* `python3 ppo_train_aliengo.py`
OR 
go to /learning_ws and 
* `./ run_ppo_training.sh`

Similarly to train with DDQN (double deep Q network) or DQN (you can change by  ddqn = true to false)
* `./ run_ddqn_training.sh`

## How to Test
go to /learning_ws and 
To test learned PPO model
* `./ run_ppo_test.sh`
To test learned DDQN model
* `./ run_ddqn_test.sh`

Note: If you want to see in Gazebo GUI go to aliengo/aliengo_gazebo/launch , open aliengo_empty_world.launch
change <arg name="gui" default="false"/> to true (we  make this false so that our computer can handle learning process in realtime)

