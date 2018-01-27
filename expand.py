from pywinauto.application import Application
from pywinauto.keyboard import SendKeys
from pywinauto.mouse import click
from os import system
from time import sleep
from itertools import tee
import re
import csv
import sys
from transforms import convert
from random import sample
from merge import merge_out
import numpy as np
from math import atan2, degrees, sqrt, sin, cos, pi
from scipy.optimize import minimize

NUM_ITERATIONS = 1


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def auto_correlate(directory, img_prefix, *positions):
    positions = [int(x) for x in positions]
    for i, j in pairwise(positions):
        with open(directory + r'/pitch1.txt', 'w') as w2:
            w2.write(str(4.5))
        with open(directory + r'/pitch2.txt', 'w') as w2:
            w2.write(str(4.5))

        with open(directory + r'/roll1.txt', 'w') as w2:
            w2.write(str((31.5 - 1.5 * i) * .095))

        with open(directory + r'/roll2.txt', 'w') as w2:
            w2.write(str((31.5 - 1.5 * j) * .095))

        print(r'copy /Y "%s\%s-%04d_0.tif" "%s\%d\%s-0000_1.tif"'
              % (directory, img_prefix, j, directory, i, img_prefix))
        system(r'copy /Y "%s\%s-%04d_0.tif" "%s\%d\%s-0000_1.tif"'
               % (directory, img_prefix, j, directory, i, img_prefix))

        with open('%s/points.csv' % directory) as f:
            points = f.readlines()

        points = [[float(x) for x in point] for point in points]
        pt = points[i] + points[j]

        transform = convert(*pt)
        transform = '%10.9e %10.9e %10.9e %10.9e %10.9e %10.9e' % transform

        with open(r'%s\%d\project.xml' % (directory, i)) as f:
            s = f.read()

        find = re.compile(r'(<camera id="1">.*?)<orientation>.*?</orientation>')

        news = find.sub(r'\1<orientation>%s</orientation>' % transform, s)

        wf = open(r'%s\%d\project_tmp.xml' % (directory, i), 'w')
        wf.write(news)
        wf.close()

        system(r'"C:\Program Files\7-Zip\7z.exe" a -tzip %s\%d\%d.z3d '
               r'%s\%d\project_tmp.xml' % (directory, i, i, directory, i))

        system(r'"C:\Program Files\7-Zip\7z.exe" d'
               r' -tzip %s\%d\%d.z3d project.xml' % (directory, i, i))

        system(r'"C:\Program Files\7-Zip\7z.exe" rn -tzip %s\%d\%d.z3d '
               r'project_tmp.xml project.xml' % (directory, i, i))

        Application().start(r'"C:\Program Files\Correlated Solutions\Vic-3D 7\Vic3D.exe"'
                            r' %s\%d\%d.z3d' % (directory, i, i))
        sleep(1.5)

        SendKeys("^R")
        SendKeys("{VK_RETURN}")
        sleep(48)

        SendKeys("{RIGHT}{VK_RETURN}^E")
        sleep(1)

        click('left', (850, 870))
        sleep(3.5)
        click('left', (1060, 870))

        SendKeys("^S^Q")
    sleep(1)


def combine(directory, img_prefix, *positions):
    positions = [int(x) for x in positions]
    for i, j in pairwise(reversed(positions[:-1])):
        with open(r'%s\%d\project.xml' % (directory, i)) as f:
            s = f.read()

        prev_data = []
        with open(r'%s\%d\%s-0000_0.csv' % (directory, j, img_prefix)) as f:
            read = csv.DictReader(f)
            for line in read:
                prev_data.append(line)

        prev_data = [{y.strip(' "\'\t\r\n'): float(x[y]) for y in x.keys()} for x in prev_data]

        next_data = []
        with open(r'%s\%d\%s-0000_0.csv' % (directory, i, img_prefix)) as f:
            read = csv.DictReader(f)
            for line in read:
                next_data.append(line)

        next_data = [{y.strip(' "\'\t\r\n'): float(x[y]) for y in x.keys()} for x in next_data]
        my = max([int(x['y']) for x in next_data])
        mx = max([int(x['x']) for x in next_data])

        table = [[None] * (my + 2) for _ in range(mx + 2)]

        for p in next_data:
            for dx in range(-2, 2):
                for dy in range(-2, 2):
                    table[int(p['x']) + dx][int(p['y']) + dy] = p

        params = [0] * 6
        for x in range(NUM_ITERATIONS):
            params_tmp = gen_params(prev_data, table)
            for y in range(len(params)):
                params[y] += params_tmp[y]

        print(params)
        params = [param / NUM_ITERATIONS for param in params]
        print(params)

        find = re.compile(r'</correlationoptions>.*?</project>', re.DOTALL)
        news = find.sub(r'</correlationoptions><coordinatetransformations lri="coordinatetransformations">'
                        r'<coordinatetransformation name="a">%.6f %.6f %.6f 0 0 0</coordinatetransformation>'
                        r'<coordinatetransformation name="z">0 0 0 %.6f %.6f %.6f</coordinatetransformation>'
                        r'</coordinatetransformations></project>' % (params[0], params[1], params[2],
                                                                     params[3], params[4], params[5]), s)

        with open(r'%s\%d\project_tmp.xml' % (directory, i), 'w') as wf:
            wf.write(news)

        system(r'"C:\Program Files\7-Zip\7z.exe" a -tzip %s\%d\%d.z3d '
               r'%s\%d\project_tmp.xml' % (directory, i, i, directory, i))

        system(r'"C:\Program Files\7-Zip\7z.exe" d'
               r' -tzip %s\%d\%d.z3d project.xml' % (directory, i, i))

        system(r'"C:\Program Files\7-Zip\7z.exe" rn -tzip %s\%d\%d.z3d '
               r'project_tmp.xml project.xml' % (directory, i, i))

        Application().start(r'"C:\Program Files\Correlated Solutions\Vic-3D 7\Vic3D.exe"'
                            r' %s\%d\%d.z3d' % (directory, i, i))

        sleep(1.5)

        click('left', (200, 35))
        SendKeys("{DOWN}{DOWN}{DOWN}{RIGHT}{VK_RETURN}")
        sleep(.5)
        SendKeys("{VK_RETURN}")
        sleep(1)
        SendKeys("{VK_RETURN}")

        click('left', (200, 35))
        SendKeys("{DOWN}{DOWN}{DOWN}{RIGHT}{VK_RETURN}")
        sleep(.5)
        SendKeys("{DOWN}{VK_RETURN}")
        sleep(1)
        SendKeys("{VK_RETURN}")
        SendKeys("^S^Q")
        merge_out(r'%s\%d\%s-0000_0.out' % (directory, j, img_prefix),
                  r'%s\%d\%s-0000_0.out' % (directory, i, img_prefix),
                  r'%s\%d\%s-0000_0.out' % (directory, j, img_prefix))


def cache(directory, img_prefix, *positions):
    positions = [int(x) for x in positions]
    for i in positions[:-1]:
        system(r'copy /Y "%s\%d\%s-0000_0.out" "%s\%d\cache.out"'
               % (directory, i, img_prefix, directory, i))


def uncache(directory, img_prefix, *positions):
    positions = [int(x) for x in positions]
    for i in positions[:-1]:
        system(r'copy /Y "%s\%d\cache.out" "%s\%d\%s-0000_0.out"'
               % (directory, i, directory, i, img_prefix))


def get_corresponding(p, table):
    if int(p['x'] + p['q']) >= len(table) or int(p['x'] + p['q']) < 0:
        return None
    if int(p['y'] + p['r']) >= len(table[0]) or int(p['y'] + p['r']) < 0:
        return None
    return table[int(p['x'] + p['q'])][int(p['y'] + p['r'])]


def suitable(p, table):
    return get_corresponding(p, table) is not None


def gen_params(prev_data, table):
    pre = []

    for x in range(6):
        for y in sample(prev_data, len(prev_data)):
            if suitable(y, table) and y not in pre:
                pre.append(y)
                break

    nex = []
    for x in pre:
        nex.append(get_corresponding(x, table))

    x_e = np.array([[pre[0]['X'], pre[1]['X'], pre[2]['X']],
                   [pre[0]['Y'], pre[1]['Y'], pre[2]['Y']],
                   [pre[0]['Z'], pre[1]['Z'], pre[2]['Z']]])

    x_o = np.array([[nex[0]['X'], nex[1]['X'], nex[2]['X']],
                   [nex[0]['Y'], nex[1]['Y'], nex[2]['Y']],
                   [nex[0]['Z'], nex[1]['Z'], nex[2]['Z']]])

    x_u = np.array([[pre[3]['X'], pre[4]['X'], pre[5]['X']],
                   [pre[3]['Y'], pre[4]['Y'], pre[5]['Y']],
                   [pre[3]['Z'], pre[4]['Z'], pre[5]['Z']]])

    x_v = np.array([[nex[3]['X'], nex[4]['X'], nex[5]['X']],
                   [nex[3]['Y'], nex[4]['Y'], nex[5]['Y']],
                   [nex[3]['Z'], nex[4]['Z'], nex[5]['Z']]])

    m = (x_e - x_u).dot(np.linalg.inv(x_o - x_v))

    for c in range(m.shape[1]):
        m[:, c] = m[:, c] / np.linalg.norm(m[:, c])

    t = x_e - m.dot(x_o)
    t = np.transpose(t)[0].tolist()

    sy = sqrt(m[0, 0]**2 + m[0, 1]**2)

    alpha = atan2(-m[1, 2], m[2, 2])
    beta = atan2(m[0, 2], sy)
    gamma = atan2(-m[0, 1], m[0, 0])

    params = [alpha, beta, gamma]
    params.extend(t)

    print('Initial guess params: %s' % str([degrees(param) for param in params[:3]] + params[3:]))

    def get_matrix(a, b, g):

        r_x = np.array([[1, 0, 0],
                        [0, cos(a), -sin(a)],
                        [0, sin(a), cos(a)]
                        ])

        r_y = np.array([[cos(b), 0, sin(b)],
                        [0, 1, 0],
                        [-sin(b), 0, cos(b)]
                        ])

        r_z = np.array([[cos(g), -sin(g), 0],
                        [sin(g), cos(g), 0],
                        [0, 0, 1]
                        ])

        r = r_z.dot(r_y).dot(r_x)

        return r

    points = []
    for point in prev_data:
        if suitable(point, table):
            points.append(point)

    def diff(p, rotate, tr):
        corr = get_corresponding(p, table)
        q = np.array([p['X'], p['Y'], p['Z']])
        r = np.array([corr['X'], corr['Y'], corr['Z']])
        d = rotate.dot(r) + tr - q
        return d.dot(d)

    def eq(p):
        a, b, g, tx, ty, tz = p
        r = get_matrix(a, b, g)
        tr = np.array([tx, ty, tz])
        ret = sum([diff(point_i, r, tr) for point_i in points]) / (len(points) + 1)
        print(ret)
        return ret

    bounds = ((-pi, pi), (-pi, pi), (-pi, pi), (None, None), (None, None), (None, None))
    solve = minimize(eq, np.array(params), bounds=bounds, method='SLSQP')
    print(solve)
    params = solve['x'].tolist()

    params = [degrees(param) for param in params[:3]] + params[3:]
    print('Solved params: %s' % params)

    return params


if __name__ == "__main__":
    auto_correlate(sys.argv[1], sys.argv[2], *sys.argv[3:])
    cache(sys.argv[1], sys.argv[2], *sys.argv[3:])
    uncache(sys.argv[1], sys.argv[2], *sys.argv[3:])
    combine(sys.argv[1], sys.argv[2], *sys.argv[3:])
