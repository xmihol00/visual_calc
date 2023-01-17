import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from const_config import WRITERS_LABELS

with open(WRITERS_LABELS, "wb") as labels_file:
    pickle.dump([
        "2378+88",
        "785+6",
        "6-7/2*3+4",
        "76+34-25",
        "4+5-6*7/8",
        "11+7-89*9",
        "46/23-35",
        "21+81",

        "4200-3500",
        "7+8-9/3",
        "47*82/41",
        "66-55+33",
        "70+80-90",
        "237/28-56",
        "6278-3014",
        "27+3-4/8",
        "25+405",
        "2+1-0",

        "435+637*924",
        "100/50*23-69",
        "1826+2325/62",

        "235+43/7",
        "785+6-8",
        "6-7+4/3-10",
        "47*8-6+20",
        "21+884-30",
        "4+51+8/3*7-1",
        "46/271-12*9",
        "34*7/9+40",
        "96-47/31+210",

        "68/3-5+22",
        "2125-8/7+44",
        "66+99-8/7",
        "47*10/2",
        "9684-43*20",
        "81-21+43",
        "12/8+5-60",
        "6+8",
        "22*1+5",

        "48+623",
        "85-102*3",
        "274+222/6",
        "789-100+2",
        "38-4+20*3",
        "78+9",
        "36+76",
        "55-88+1",
        "66*28-4",

        "51*3+12",
        "6-4/77",
        "892+3-2/4",
        "76/148+5",
        "3*4*2/7",
        "41-32+8",
        "65*21/9",

        "13+5*92",
        "4-68/43",
        "27*456/1348",
        "35+89+4-9",
        "66/713*42",
        "54-32*8",
        "781+953",
    ], 
    labels_file, protocol=pickle.HIGHEST_PROTOCOL)
