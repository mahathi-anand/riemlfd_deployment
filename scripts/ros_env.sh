# Drop the host ROS 2 install (Kilted, sourced in ~/.bashrc) so only this env's Jazzy is used
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ":" "\n" | grep -v "^/opt/ros" | paste -sd:)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ":" "\n" | grep -v "^/opt/ros" | paste -sd:)
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_LOCALHOST_ONLY

# Same middleware settings as the CRISP MuJoCo container (RMW=cyclone, domain 100, loopback).
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=100
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_NETWORK_INTERFACE=lo
export CYCLONEDDS_URI=file://$PIXI_PROJECT_ROOT/config/cyclone_config.xml
