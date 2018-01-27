from numpy import dot, cross, array
from numpy.linalg import norm, inv
from math import pi, sin, cos, atan2, sqrt
from re import split


def transform(points, out):
    f_in = open(points, 'rt')
    f_out = open(out, 'wt')

    nums = [float(x) for x in split(',\W*?', f_in.readline())]

    i_x = nums[0]
    i_y = nums[1]
    i_z = nums[2]
    i_p = nums[3]
    i_r = nums[4]

    this_line = f_in.readline()
    while this_line:
        nums = [float(x) for x in split(',\W*?', this_line)]

        rx, ry, rz, tx, ty, tz = convert(i_x, i_y, i_z, i_p, i_r, nums[0], nums[1], nums[2], nums[3], nums[4])

        f_out.write('%10.9e %10.9e %10.9e %10.9e %10.9e %10.9e\n' % (rx, ry, rz, tx, ty, tz))
        this_line = f_in.readline()

    f_in.close()
    f_out.close()


def convert(x, y, z, pitch, roll, xt, yt, zt, pitch_t, roll_t):
    # Initial RoboCIM Cartesian Coordinates
    init = array([x, y, z])

    # RoboCIM Cartesian Coordinates after movement
    trans = array([xt, yt, zt])

    # The displacement of the sensor, horizontally and vertically in
    # millimeters, from the root of the end effector
    dh = -47
    dv = 41

    p_off = 3.5
    with open('roll1.txt') as f:
        r_off1 = float(f.read())
    with open('roll2.txt') as f:
        r_off2 = float(f.read())

    # Convert to radians
    pitch = (pitch + p_off) / 180 * pi
    roll = (roll + r_off1) / 180 * pi
    pitch_t = (pitch_t + p_off) / 180 * pi
    roll_t = (roll_t + r_off2) / 180 * pi

    # Find flat direction of the arm
    xyn = array([x, y, 0])
    xyn = xyn / norm(xyn)
    xn = xyn[0]
    yn = xyn[1]

    # Construct end effector direction
    ebt = array([xn * sin(pitch),
                 yn * sin(pitch),
                 cos(pitch)])

    # Find flat direction of the arm
    xyn = array([xt, yt, 0])
    xyn = xyn / norm(xyn)
    xn = xyn[0]
    yn = xyn[1]

    # Construct end effector direction
    e = array([xn * sin(pitch_t),
               yn * sin(pitch_t),
               cos(pitch_t)])

    # Construct a basis for a coordinate with the initial arm direction as the
    # x axis in RoboCIM coordinates
    b1 = array([[x, y, 0] / norm([x, y, 0]), cross([0, 0, 1], [x, y, 0] / norm([x, y, 0])), [0, 0, 1]])
    b1 = b1.transpose()

    # Construct basis with end effector as z axis in B1 coordinates
    b2 = array([cross([0, 1, 0], [-sin(pitch), 0, -cos(pitch)]), [0, 1, 0], [-sin(pitch), 0, -cos(pitch)]])
    b2 = b2.transpose()

    # Construct displacement to and direction of sensor in B2 coordinates
    displacement_init = array([dh * cos(-roll + pi / 2), dh * sin(-roll + pi / 2), dv])
    direction_init = array([dh * cos(-roll - pi / 2), dh * sin(-roll - pi / 2), 0])

    # Convert to RoboCIM coordinates
    displacement_init_r = b1.dot(b2).dot(displacement_init)
    direction_init_r = b1.dot(b2).dot(direction_init)

    # Construct a basis for a coordinate with the final arm direction as the
    # x axis in RoboCIM coordinates
    b3 = array([[xt, yt, 0] / norm([xt, yt, 0]), cross([0, 0, 1], [xt, yt, 0] / norm([xt, yt, 0])), [0, 0, 1]])
    b3 = b3.transpose()

    # Construct basis with end effector as z axis in B3 coordinates
    b4 = array([cross([0, 1, 0], [-sin(pitch_t), 0, -cos(pitch_t)]), [0, 1, 0], [-sin(pitch_t), 0, -cos(pitch_t)]])
    b4 = b4.transpose()

    # Construct displacement to sensor in B4 coordinates
    displacement_final = array([dh * cos(-roll_t + pi / 2), dh * sin(-roll_t + pi / 2), dv])
    direction_final = array([dh * cos(-roll_t - pi / 2), dh * sin(-roll_t - pi / 2), 0])

    # Convert to RoboCIM coordinates
    displacement_final_r = b3.dot(b4).dot(displacement_final)
    direction_final_r = b3.dot(b4).dot(direction_final)

    # Calculate total displacement
    move = (trans + displacement_final_r) - (init + displacement_init_r)

    # Use end effector orientation and optical axis to construct a basis for
    # the VIC-3D coordinate system
    by = -e
    bz = direction_final_r / norm(direction_final_r)
    bx = -cross(by, bz)
    b = array([bx, by, bz])
    b = b.transpose()

    # Convert to VIC-3D coordinates
    trans_f = dot(inv(b), move)

    direction_init_normalized = direction_init_r / norm(direction_init_r)
    ini_mat = array([-cross(-ebt, direction_init_normalized), -ebt, direction_init_normalized])
    ini_mat = ini_mat.transpose()

    r = dot(inv(b), ini_mat)

    # Calculate roll, yaw, and pitch respectively
    sy = sqrt(r[0, 0] ** 2 + r[0, 1] ** 2)

    alpha = atan2(-r[1, 2], r[2, 2])
    beta = atan2(r[0, 2], sy)
    gamma = atan2(-r[0, 1], r[0, 0])

    alpha = alpha / pi * 180
    beta = beta / pi * 180
    gamma = gamma / pi * 180

    # Print out necessary parameters
    tx = trans_f[0]
    ty = trans_f[1]
    tz = trans_f[2]
    return alpha, beta, gamma, tx, ty, tz
