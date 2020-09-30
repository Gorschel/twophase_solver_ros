#!/usr/bin/env python
# coding: utf-8
# name can be changed back to solver_client.py if 

import sys
import rospy
from twophase_solver_ros.srv import Solver
from cube_scan_opencv import scan_cube

def cube_solver_client(cube):
    rospy.wait_for_service('cube_solver')
    try:
        cube_solver = rospy.ServiceProxy('cube_solver', Solver)
        resp = cube_solver(cube)
        return resp
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    retval, cube = scan_cube() # vlt return val nötig um roboter selbst die seiten wenden zu lassen
    print "\nscanned cube: %s" %cube    # 'DLRRUUDFDBBBDRBLLBRRLFFBRFDFUFDDULULFDBBLDFLUUFRLBRURU' # example CubeDefString 
    resp = cube_solver_client(cube)     # find solution for cube
    if resp.movecount == 0:
        print "\n%s" %(resp.solution)
        pass
    else:
        print "received solution: %s (%s moves)\n" %(resp.solution, resp.movecount)
        pass