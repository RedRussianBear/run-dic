import struct
import sys
from collections import OrderedDict


def merge_out(output, *source):
    if len(source) < 1:
        return

    header, space1, space2, space25, space3, space4 = '', '', '', '', '', ''
    data = []
    keys = []
    var_to_long = OrderedDict()
    var_head = OrderedDict()

    for i in source:
        with open(i, 'rb') as datum:
            header = datum.read(23)
            datum.read(ord(datum.read(1))).decode('utf-16-be')  # filename
            space1 = datum.read(7)
            filename = datum.read(ord(datum.read(1))).decode('utf-16-be')
            space2 = datum.read(24)
            datum.read(4)  # h
            space25 = datum.read(4)
            datum.read(4)  # w

            var_vals = {'filename': filename}

            while True:
                b = datum.read(19)
                if not b:
                    break
                var_header = b
                var_name = datum.read(ord(datum.read(1))).decode('utf-16-be')
                if var_name not in keys:
                    keys.append(var_name)

                space3 = datum.read(7)
                var_name_long = datum.read(ord(datum.read(1))).decode('utf-16-be')
                space4 = datum.read(4)

                var_to_long[var_name] = var_name_long
                var_head[var_name] = var_header

                data_len = struct.unpack('>i', datum.read(4))[0]
                col = list(struct.unpack('<%df' % data_len, datum.read(4 * data_len)))

                var_vals[var_name] = col

            data.append(var_vals)

    combined = {}
    for key in keys:
        combined[key] = []

    for datum in data:
        for key in keys:
            combined[key].extend(datum[key])

    ux = set(combined['x'])
    w = len(ux)
    h = 0

    combined = {}
    for key in keys:
        combined[key] = []

    for datum in data:
        print(datum['filename'])
        points = [{key: datum[key][x] for key in keys} for x in range(len(datum['x']))]
        new_thing = []

        while len(points) > 0:
            y = points[0]['y']
            h += 1
            unused = []
            for x in sorted(ux):
                if len(points) > 0 and points[0]['x'] == x and points[0]['y'] == y:
                    new_thing.append(points.pop(0))
                else:
                    unused.append(x)

            for x in range(len(unused)):
                new_dict = {key: 0 for key in keys}
                new_dict['y'] = y
                new_dict['x'] = max(ux) + 1 + x
                new_dict['sigma'] = -1
                new_thing.append(new_dict)

        points = new_thing
        normalized = {key: [point[key] for point in points] for key in keys}
        for key in keys:
            combined[key].extend(normalized[key])

    print(h)
    print(w)
    print(h * w)
    print(len(combined['X']))

    with open(output, 'wb') as f:
        filename = source[-1].replace('.out', '.tif').encode('utf-16-be')
        f.write(header)
        f.write(struct.pack('B', len(filename)))
        f.write(filename)
        f.write(space1)
        f.write(struct.pack('B', len(filename)))
        f.write(filename)
        f.write(space2)
        f.write(struct.pack('>i', h))
        f.write(space25)
        f.write(struct.pack('>i', w))

        for key in keys:
            f.write(var_head[key])
            var_name = key.encode('utf-16-be')
            f.write(struct.pack('B', len(var_name)))
            f.write(var_name)
            f.write(space3)
            var_name_long = var_to_long[key].encode('utf-16-be')
            f.write(struct.pack('B', len(var_name_long)))
            f.write(var_name_long)
            f.write(space4)

            data_len = struct.pack('>i', len(combined[key]))
            datum = struct.pack('<%df' % len(combined[key]), *(tuple(combined[key])))
            f.write(data_len)
            f.write(datum)


if __name__ == "__main__":
    merge_out(sys.argv[1], *sys.argv[2:])
