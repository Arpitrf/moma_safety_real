pal_navigation_cfg_tiago_dual


rosrun rqt_reconfigure rqt_reconfigure
# Changes made using rqt_reconfigure:
/move_base/global_costmap/inflation_layer: 
1. Toggled off "enabled"
2. cost_scaling_factor: 25.0 to 0.0
3. inflation_Radius from 0.6 to 0.0


planner_frequency 2.0 to 20.0


Installation of grounded sam 2:
1. pip install -e . and 
2. pip install grounding dino (command in repo) 
3. pip install other packages