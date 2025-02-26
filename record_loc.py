from pygba import PyGBA
from env_setup import *
import json

locations = {
    1: [3, 4, 7, 6], # Intro
    2: [2, 3, 4, 5, 6, 7], # Flizard
    3: [2, 3, 4, 5, 6, 7, 8, 9], # Childre
    4: [2, 3, 4, 5, 6], # Hellbat
    5: [2, 3, 5, 6, 7], # Mageisk
    6: [2, 3, 5, 6], # Baby Elves 1
    7: [1, 3, 4, 5], # Anubis
    8: [1, 3, 4, 5], # Hanumachine
    9: [1, 3, 4, 5], # Blizzack
    10: [1, 3, 4, 5], # Copy X
    11: [2, 3, 4, 5, 6], # Foxtar
    12: [2, 3, 4, 5], # le Cactank
    13: [2, 3, 5, 6], # Volteel
    14: [1, 3, 4, 5, 6], # Kelverian
    15: [1, 3, 4, 6, 7, 8], # Sub Arcadia
    16: [1, 3, 5, 6, 8, 9] # Final
}

positions = {}

gba = PyGBA.load(GBA_ROM, GBA_SAV)

for stage, chkpts in locations.items():
    for chkpt in chkpts:
        gba.core.reset()
        gba.wait(10)
        gba.press_start(10)
        gba.wait(10)
        gba.press_start(10)
        gba.wait(10)
        gba.press_start(10)
        gba.wait(10)
        gba.press_start(10)
        gba.wait(10)
        if gba.read_u16(0x02030318) == 18434 and gba.read_u16(0x0203031C) == 18442:
            gba.press_up(10)
            gba.wait(30)
            if gba.read_u16(0x02030318) == 18426 and gba.read_u16(0x0203031C) == 18450:
                gba.press_a(10)
                gba.wait(100)
                gba.press_start(10)
                gba.wait(30)
                gba.core.memory.u8.raw_write(0x0202FE60, stage)
                gba.core.memory.u8.raw_write(0x0202FE62, chkpt)
                gba.press_start(10)
                gba.wait(30)
                x = gba.core.memory.u32.raw_read(0x02037CB4)
                y = gba.core.memory.u32.raw_read(0x02037CB8)
                if stage not in positions:
                    positions[stage] = {}
                positions[stage][chkpt] = (x, y)


with open('positions.json', 'w', encoding='utf-8') as f:
    json.dump(positions, f, indent = 2)
