#!/usr/bin/env python
# coding: utf-8

import rospy
from solver import solve # actual twophase solver algorithm
from twophase_solver_ros.srv import Solver, SolverResponse
from std_msgs.msg import String

def check_solution(solStr):
    """check for errors and split movecount from solution string"""
    if solStr[len(solStr)-2 : ] == 'f)':
        print "Solution: %s" %(solStr)
        movecount = int(solStr[solStr.find('(')+1 : len(solStr)-2])
        solution = solStr[ : len(solStr)-6]
    else:
        movecount = 0
        solution = solStr + '\nMaybe the cube faces were not oriented correctly.'
    return solution, movecount

def handle_solve(req):
    print "\nreceived problem: %s" %(req.defstr)
    solStr = solve(str(req.defstr)) # twophase algorithm
    solution, movecount = check_solution(solStr) # check for errors
    return SolverResponse(movecount, solution)

if __name__ == "__main__":
    rospy.init_node('cube_solver_server')
    s = rospy.Service('cube_solver', Solver, handle_solve)
    print "Tables loaded. Ready to solve a cube."
    rospy.spin()