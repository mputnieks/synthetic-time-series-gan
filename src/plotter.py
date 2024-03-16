import matplotlib.pyplot as plt
import numpy as np


def apply_cgm_plt_style(plt, title):
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Glucose, mmol/L", fontsize=12)
    plt.ylim(0, 21)
    plt.xlim(-0.5, 47.5)
    
    # Set x-ticks at every two hours starting from 00:00
    x_ticks_positions = np.arange(-0.5, 49.5, step=4)
    x_ticks_labels = [f"{hour:02d}:{minute:02d}" for hour in range(0, 25, 2) for minute in [0]]
    plt.xticks(x_ticks_positions, x_ticks_labels)

    # Add vertical gridlines at every two hours
    plt.grid(axis='x', linestyle='-', alpha=0.7)


def plot_glucose(data, rows, title, labels=None):
    plt.figure(figsize=(10, 4))
    apply_cgm_plt_style(plt, title)
    
    for row in rows:
        cgm_data = data.iloc[row].iloc[:48] 
        # if has labels, use them, if not, generate default
        if labels:
            plt.plot(cgm_data, label=labels[row]+ ", " + str(data.iloc[row].iloc[48]))
        else:
            plt.plot(cgm_data, label="Segment " + str(row) + ", " + str(data.iloc[row].iloc[48]))
    
    plt.legend(loc='upper right')
    plt.show()


def plot_glucose_steps(data, row, title, label=None):
    plt.figure(figsize=(10, 4))
    apply_cgm_plt_style(plt, title)
    
    # CGM
    cgm_data = data.loc[row].iloc[:48] 
    if label: # if has label, use it, if not, generate default
        plt.plot(cgm_data, label=label)
    else:
        plt.plot(cgm_data, label="Glucose Segment " + str(row) + ", " + str(data.loc[row].iloc[48]))
    plt.legend(loc='upper right')
    
    # Steps
    steps = data.loc[row].iloc[49:len(data.loc[row])] # get steps data
    axes2 = plt.twinx()
    axes2.set_ylabel('Steps', fontsize=12)
    for i in range(len(steps)):
        axes2.bar(i, steps[i], color='blue', alpha=0.5, width=0.9, align='center')   
    # arrange axis ticks
    axes2.set_ylim(0, 1200)
    axes2.set_yticks(np.arange(0, 1300, 100))
    axes2.set_yticklabels(np.arange(0, 1300, 100))
    plt.show()


def plot_covariate_weekday(data, title, ymin, ymax):
    plt.figure(figsize=(8, 5))
    apply_cgm_plt_style(plt, title)
    
    # crop data to the first 48 columns
    data = data.iloc[:, :49]
    # group data by weekday and calculate the mean value of each of the first 48 columns
    data = data.groupby(['week_day']).mean().iloc[:, :48]
    
    print(data)
    
    # plot each row
    for row in range(len(data)):
        plt.plot(data.iloc[row], label=data.index[row])
    
    plt.legend(loc='lower right')
    plt.ylim(ymin, ymax)
    plt.show()