import numpy as np
import matplotlib.pylab as plt

data = np.load('../data/digits_and_characters_1/CompleteDataSet_testing_tuples.npy', allow_pickle=True)

for label, name in [("0", "zeros"), ("1", "ones"), ("2", "twos"), ("3", "threes"), ("4", "fours"), ("5", "fives"), ("6", "sixes"),
                    ("7", "sevens"), ("8", "eights"), ("9", "nines"), ("+", "pluses"), ("-", "minuses"), ("*", "astrics"), ("%", "slashes")]:
    samples = data[np.where(data[:, 1] == label)]
    np.save(f"../data/separated_characters/{name}", samples)



#ones = np.load("../data/separated_characters/ones.npy", allow_pickle=True)
#for sample in ones:
#    fig, ax = plt.subplots(1,1)
#    ax.imshow(sample[0], cmap='gray')
#    ax.set_title(sample[1])
#    plt.show()
