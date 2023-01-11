#!/usr/bin/env python
# coding: utf-8

import rospy
from clean import scan_cube
from twophase_solver.solver import solve  # actual twophase solver algorithm
from twophase_solver_ros.srv import Solver, SolverResponse


def check_solution(solStr):
    """check for errors and split movecount from solution string"""
    print('checking solution: {}'.format(solStr))
    if '(0f)' in solStr:
        print('no solution')
        solution = solStr
        movecount = 0
    elif solStr[len(solStr) - 2:] == 'f)':
        print "\nSolution found: %s" % (solStr)
        movecount = int(solStr[solStr.find('(') + 1: len(solStr) - 2])
        solution = solStr[: len(solStr) - 6]
    else:
        movecount = 0
        solution = solStr + '\nMaybe the cube faces were not oriented correctly.'
    return solution, movecount


def handle_solve(req):
    print "\nstarting image processing.."
    retval, cube_def_str = scan_cube()
    print "\ncube-state scanned: {}".format(cube_def_str)
    print "\nfinding solution.."
    sol_str = solve(cubestring=cube_def_str, max_length=1000, timeout=50)  # twophase algorithm
    if sol_str == 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB':
        return SolverResponse(0, '')
    solution, movecount = check_solution(sol_str)  # check for errors
    #print('response: {} {}'.format(movecount, solution))
    return SolverResponse(movecount, solution)


if __name__ == "__main__":
    rospy.init_node('cube_scan_solver_server')
    s = rospy.Service('cube_resolutor', Solver, handle_solve)
    print "Tables loaded. Ready to solve a cube."
    rospy.spin()
