import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

def show_map(heatmap, title = "", show_text=False, normalized = True):
        num_rounds = heatmap.shape[1]
        num_clients = heatmap.shape[0]
        fig, ax = plt.subplots()
        if normalized:
        	final_map = heatmap / LA.norm(heatmap, axis=0)
        else:
        	final_map = heatmap
        #print(self.map)
        #print(LA.norm(self.map, axis=0))
        im = ax.imshow(final_map)
        ax.set_xticks(np.arange(heatmap.shape[1]))
        ax.set_yticks(np.arange(heatmap.shape[0]))
        xlabels = np.arange(num_rounds)
        ylabels = ["client" + str(i) for i in range(num_clients)]
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

        if show_text:
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    text = ax.text(j,i, round(final_map[i,j], 2), ha="center", va="center", color="b")
            
        #print(LA.norm(self.map))
        plt.xlabel("rounds")
        plt.ylabel("clients")
        plt.title(title)
        plt.colorbar(im, shrink=0.5)
        #plt.show()
        return  fig, ax
        
def plot_range(data, line='-', color=None):
    x = np.arange(data.shape[1])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if color==None:
        ax,  = plt.plot(x, mean,line)
        plt.fill_between(x, mean-std, mean+std, alpha = 0.5)
    else:
        ax,  = plt.plot(x, mean,line, color=color)
        plt.fill_between(x, mean-std, mean+std, alpha = 0.5, color=color)
    return ax
