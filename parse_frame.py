#!/usr/bin/env python3
"""Parse DJI DroneID FRAME hex output (from remove_turbo) to JSON.

Usage:
    python parse_frame.py <hex_string>
    python parse_frame.py  # reads from stdin

Example:
    python parse_frame.py 581002a200f71f334e5a434a364e303234314c30460000...
"""

import struct
import json
import sys
import argparse

DRONEID_MAX_LEN = 91

DRONEID_DRONE_TYPES = {
    "1": "Inspire 1",
    "2": "Phantom 3 Series",
    "3": "Phantom 3 Series",
    "4": "Phantom 3 Std",
    "5": "M100",
    "6": "ACEONE",
    "7": "WKM",
    "8": "NAZA",
    "9": "A2",
    "10": "A3",
    "11": "Phantom 4",
    "12": "MG1",
    "14": "M600",
    "15": "Phantom 3 4k",
    "16": "Mavic Pro",
    "17": "Inspire 2",
    "18": "Phantom 4 Pro",
    "20": "N2",
    "21": "Spark",
    "23": "M600 Pro",
    "24": "Mavic Air",
    "25": "M200",
    "26": "Phantom 4 Series",
    "27": "Phantom 4 Adv",
    "28": "M210",
    "30": "M210RTK",
    "31": "A3_AG",
    "32": "MG2",
    "34": "MG1A",
    "35": "Phantom 4 RTK",
    "36": "Phantom 4 Pro V2.0",
    "38": "MG1P",
    "40": "MG1P-RTK",
    "41": "Mavic 2",
    "44": "M200 V2 Series",
    "51": "Mavic 2 Enterprise",
    "53": "Mavic Mini",
    "58": "Mavic Air 2",
    "59": "P4M",
    "60": "M300 RTK",
    "61": "DJI FPV",
    "63": "Mini 2",
    "64": "AGRAS T10",
    "65": "AGRAS T30",
    "66": "Air 2S",
    "67": "M30",
    "68": "DJI Mavic 3",
    "69": "Mavic 2 Enterprise Advanced",
    "70": "Mini SE",
}

CRC_INIT = 0x3692
CRC_POLY = 0x11021


def _calc_crc(data: bytes) -> str:
    try:
        import crcmod
        fn = crcmod.mkCrcFun(CRC_POLY, initCrc=CRC_INIT, rev=True)
        return "%04x" % fn(data)
    except ImportError:
        # CRC-16/CCITT reversed (poly 0x8408 = bit-reverse of 0x1021)
        crc = CRC_INIT
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
        return "%04x" % crc


def parse_frame(hex_str: str) -> dict:
    raw = bytes.fromhex(hex_str.strip())

    if len(raw) < DRONEID_MAX_LEN:
        raise ValueError(
            f"Frame too short: {len(raw)} bytes, need at least {DRONEID_MAX_LEN}"
        )

    # struct layout (91 bytes total):
    #   B   pkt_len
    #   B   unk
    #   B   version
    #   H   sequence_number
    #   H   state_info
    #   16s serial_number
    #   i   longitude  (raw / 174533.0 → degrees)
    #   i   latitude   (raw / 174533.0 → degrees)
    #   h   altitude   (raw / 3.281 → meters)
    #   h   height     (raw / 3.281 → meters)
    #   h   v_north
    #   h   v_east
    #   h   v_up
    #   h   d_1_angle
    #   Q   gps_time
    #   i   app_lat    (operator latitude)
    #   i   app_lon    (operator longitude)
    #   i   longitude_home
    #   i   latitude_home
    #   B   device_type
    #   B   uuid_len
    #   20s uuid
    #   H   crc
    f = struct.unpack("<BBBHH16siihhhhhhQiiiiBB20sH", raw[:DRONEID_MAX_LEN])

    si = f[4]
    state = {
        "alt_valid":        bool((si >> 15) & 1),
        "gps_valid":        bool((si >> 14) & 1),
        "in_air":           bool((si >> 13) & 1),
        "motors_on":        bool((si >> 12) & 1),
        "uuid_set":         bool((si >> 11) & 1),
        "home_set":         bool((si >> 10) & 1),
        "private_disabled": bool((si >> 9)  & 1),
        "serial_valid":     bool((si >> 8)  & 1),
        "veloc_z_valid":    bool((si >> 2)  & 1),
        "veloc_y_valid":    bool((si >> 1)  & 1),
    }

    crc_packet     = "%04x" % f[22]
    crc_calculated = _calc_crc(raw[:DRONEID_MAX_LEN - 2])

    result = {
        "pkt_len":         f[0],
        "unk":             f[1],
        "version":         f[2],
        "sequence_number": f[3],
        "state_info":      "0x%04x" % f[4],
        "state":           state,
        "serial_number":   f[5].decode("latin-1").rstrip("\x00"),
        "longitude":       round(f[6]  / 174533.0, 7),
        "latitude":        round(f[7]  / 174533.0, 7),
        "altitude_m":      round(f[8]  / 3.281, 2),
        "height_m":        round(f[9]  / 3.281, 2),
        "v_north":         f[10],
        "v_east":          f[11],
        "v_up":            f[12],
        "d_1_angle":       f[13],
        "gps_time":        f[14],
        "app_lat":         round(f[15] / 174533.0, 7),
        "app_lon":         round(f[16] / 174533.0, 7),
        "longitude_home":  round(f[17] / 174533.0, 7),
        "latitude_home":   round(f[18] / 174533.0, 7),
        "device_type":     DRONEID_DRONE_TYPES.get(str(f[19]), f"Unknown({f[19]})"),
        "uuid_len":        f[20],
        "uuid":            f[21].decode("latin-1").rstrip("\x00"),
        "crc_packet":      crc_packet,
        "crc_calculated":  crc_calculated,
        "crc_valid":       crc_packet == crc_calculated,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse DJI DroneID FRAME hex string to JSON"
    )
    parser.add_argument(
        "frame",
        nargs="?",
        help="Hex FRAME string from remove_turbo output (reads stdin if omitted)",
    )
    args = parser.parse_args()

    hex_str = args.frame if args.frame else sys.stdin.read()
    # strip leading "FRAME: " prefix if present
    hex_str = hex_str.strip()
    if hex_str.upper().startswith("FRAME:"):
        hex_str = hex_str.split(":", 1)[1].strip()

    result = parse_frame(hex_str)
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
