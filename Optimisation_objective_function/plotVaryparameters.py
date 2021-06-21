from cnn_split_optimize import main_function
import matplotlib.pyplot as plt
import numpy as np


settings = {
    "layers":100,
    "kernel_w":3,
    "kernel_h":3,
    "memory_size":2048,
    "cpu_size":1500,
    "server_cpu":2800,
    "edge_cores":2,
    "server_cores":4,
    "bandwidth":1000
}




def plot_changing_layers(settings):

    print("[PLOTTING GRAPHS CHANGING THE NUMBER OF LAYERS OF THE NETWORK]")
    split_list = []
    latency_list  = []
    memory_list = []
    
    for i in range(10,101):
        settings['layers'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()

    ax1.plot(*zip(*split_list) , linestyle='dotted',alpha=0.8)
    ax1.set_xticks(np.arange(10, 101 , 2.0))
    ax1.set_yticks(np.arange(0, 40 , 2.0))
    ax1.set_title("Split Point Layer vs Number of Layers")
    ax1.set_xlabel("Total Layers")
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax2.plot(*zip(*latency_list) , linestyle='dotted',alpha=0.8)
    ax2.set_xticks(np.arange(10, 101 , 2.0))
    ax2.set_title("Latency at split layer vs Number of Layers")
    ax2.set_xlabel("Total Layers")
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.plot(*zip(*memory_list),linestyle='dotted' , alpha=0.8)
    ax3.set_xticks(np.arange(10, 101 , 2.0))
    ax3.set_title("Memory at split layer vs Number of Layers")
    ax3.set_xlabel("Total Layers")
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    fig1.savefig('split-point-no-of-layers.png')
    fig2.savefig('Latency-no-of-layers.png')
    fig3.savefig('Memory-no-of-layers.png')

def plot_changing_kernel_size(settings):
    print("[PLOTTING GRAPHS CHANGING THE KERNEL SIZE OF THE NETWORK KEEPING NUMBER OF LAYERS SAME]")
    split_list = []
    latency_list  = []
    memory_list = []
    for i in range(3 , 17 , 2):
        settings['kernel_w'] = i
        settings['kernel_h'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()
    

    ax1.plot(*zip(*split_list) , linestyle='dotted',alpha=0.8)
    ax1.set_xticks(np.arange(3, 17 , 2.0))
    ax1.set_title("Split Point Layer vs Kernel Size")
    ax1.set_xlabel("Kernel Size")
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax2.plot(*zip(*latency_list) , linestyle='dotted',alpha=0.8)
    ax2.set_xticks(np.arange(3, 17 , 2.0))
    ax2.set_title("Latency at split layer vs Kernel Size")
    ax2.set_xlabel("Kernel Size")
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.plot(*zip(*memory_list) ,linestyle='dotted' ,alpha=0.8)
    ax3.set_xticks(np.arange(3, 17 , 2.0))
    ax3.set_title("Memory at split layer vs Kernel Size")
    ax3.set_xlabel("Kernel Size")
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    
    fig1.savefig('split-point-kernel-size.png')
    fig2.savefig('Latency-kernel-size.png')
    fig3.savefig('Memory-kernel-size.png')

def plot_changing_bandwidth(settings):
    print("[PLOTTING GRAPHS CHANGING THE BANDWIDTH OF THE NETWORK KEEPING NUMBER OF LAYERS SAME]")
    split_list = []
    latency_list  = []
    memory_list = []
    #WOULD TRY THIS
    for i in range(100 , 2100 , 100):
        settings['bandwidth'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()
    

    ax1.scatter(*zip(*split_list) , alpha=0.8)
    ax1.set_xticks(np.arange(100, 2100 , 100.0))
    ax1.set_title("Split Point Layer vs Bandwidth")
    ax1.set_xlabel("Bandwidth")
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax2.scatter(*zip(*latency_list) , alpha=0.8)
    ax2.set_xticks(np.arange(100, 2100 , 100.0))
    ax2.set_title("Latency at split layer vs Bandwidth")
    ax2.set_xlabel("Bandwidth")
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.scatter(*zip(*memory_list) , alpha=0.8)
    ax3.set_xticks(np.arange(100, 2100 , 100.0))
    ax3.set_title("Memory at split layer vs Bandwidth")
    ax3.set_xlabel("Bandwidth")
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    
    fig1.savefig('split-point-bandwidth.png')
    fig2.savefig('Latency-bandwidth.png')
    fig3.savefig('Memory-bandwidth.png')

def plot_changing_edge_memory(settings):
    print("[PLOTTING GRAPHS CHANGING THE TOTAL MEMORY AT EDGE KEEPING NUMBER OF LAYERS SAME]")
    split_list = []
    latency_list  = []
    memory_list = []
    memory = [256 , 512 , 1024 , 2048 , 4096 , 8192 , 16384 , 32768]
    for i in memory:
        settings['memory_size'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()
    

    ax1.plot(*zip(*split_list) ,  linestyle='dotted',alpha=0.8)
    ax1.set_title("Split Point Layer vs Edge Memory")
    ax1.set_xlabel("Total Memory")
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax2.plot(*zip(*latency_list),linestyle='dotted' , alpha=0.8)
    ax2.set_title("Latency at split layer vs Memory")
    ax2.set_xlabel("Total Memory")
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.plot(*zip(*memory_list),linestyle='dotted' , alpha=0.8)
    ax3.set_title("Memory at split layer vs Bandwidth")
    ax3.set_xlabel("Total Memory")
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    
    fig1.savefig('split-point-total-memory.png')
    fig2.savefig('Latency-total-memory.png')
    fig3.savefig('Memory-total-memory.png')


def plot_changing_edge_computation(settings):
    print("[PLOTTING GRAPHS CHANGING THE EDGE COMPUTATION SPEED KEEPING NUMBER OF LAYERS SAME]")
    split_list = []
    latency_list  = []
    memory_list = []
    for i in range(1000 , 4100 , 100):
        settings['cpu_size'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()


    ax1.plot(*zip(*split_list) ,  linestyle='dotted',alpha=0.8)
    ax1.set_xticks(np.arange(1000, 4100, 100.0))
    ax1.set_title("Split Point Layer vs CPU Speed")
    ax1.set_xlabel("CPU speed at Edge")
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax2.plot(*zip(*latency_list),linestyle='dotted' , alpha=0.8)
    ax2.set_xticks(np.arange(1000, 4100, 100.0))
    ax2.set_title("Latency at split layer vs CPU Speed")
    ax2.set_xlabel("CPU speed at edge")
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.plot(*zip(*memory_list),linestyle='dotted' , alpha=0.8)
    ax3.set_xticks(np.arange(1000, 4100, 100.0))
    ax3.set_title("Memory at split layer vs CPU Speed")
    ax3.set_xlabel("CPU speed at edge")
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    
    fig1.savefig('split-point-cpu-speed.png')
    fig2.savefig('Latency-cpu-speed.png')
    fig3.savefig('Memory-cpu-speed.png')

def plot_changing_server_computation(settings):
    print("[PLOTTING GRAPHS CHANGING THE SERVER COMPUTATION SPEED KEEPING NUMBER OF LAYERS SAME]")
    split_list = []
    latency_list  = []
    memory_list = []
    for i in range(2000 , 5100 , 100):
        settings['server_cpu'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()


    ax1.plot(*zip(*split_list) ,  linestyle='dotted',alpha=0.8)
    ax1.set_xticks(np.arange(2000, 5100, 100.0))
    ax1.set_title("Split Point Layer vs CPU Speed at Server")
    ax1.set_xlabel("CPU speed at Server")
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax2.plot(*zip(*latency_list),linestyle='dotted' , alpha=0.8)
    ax2.set_xticks(np.arange(2000, 5100, 100.0))
    ax2.set_title("Latency at split layer vs CPU Speed at Server")
    ax2.set_xlabel("CPU speed at Server")
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.plot(*zip(*memory_list),linestyle='dotted' , alpha=0.8)
    ax3.set_xticks(np.arange(2000, 5100, 100.0))
    ax3.set_title("Memory at split layer vs CPU Speed at Server")
    ax3.set_xlabel("CPU speed at Server")
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    
    fig1.savefig('split-point-cpu-speed-at-server.png')
    fig2.savefig('Latency-cpu-speed-at-server.png')
    fig3.savefig('Memory-cpu-speed-at-server.png')



def plot_changing_edge_cores(settings):
    print("[PLOTTING GRAPHS CHANGING EDGE CORES KEEPING NUMBER OF LAYERS SAME]")
    split_list = []
    latency_list  = []
    memory_list = []
    for i in range(1 , 9):
        settings['edge_cores'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()
    

    ax1.plot(*zip(*split_list) , linestyle='dotted',alpha=0.8)
    ax1.set_xticks(np.arange(1, 9))
    ax1.set_title("Split Point Layer vs Edge Cores")
    ax1.set_xlabel("Edge Cores")
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax2.plot(*zip(*latency_list) , linestyle='dotted',alpha=0.8)
    ax2.set_xticks(np.arange(1,9))
    ax2.set_title("Latency at split layer vs Edge Cores")
    ax2.set_xlabel("Edge Cores")
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.plot(*zip(*memory_list) ,linestyle='dotted' ,alpha=0.8)
    ax3.set_xticks(np.arange(1,9))
    ax3.set_title("Memory at split layer vs Edge Cores")
    ax3.set_xlabel("Edge Cores")
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    
    fig1.savefig('split-point-edge-cores.png')
    fig2.savefig('Latency-edge-cores.png')
    fig3.savefig('Memory-edge-cores.png')

def plot_changing_server_cores(settings):
    print("[PLOTTING GRAPHS CHANGING SERVER CORES KEEPING NUMBER OF LAYERS SAME]")
    split_list = []
    latency_list  = []
    memory_list = []
    for i in range(4 , 19):
        settings['server_cores'] = i
        latency , memory , edge_layer , _ = main_function(settings)
        split_list.append((i , edge_layer))
        latency_list.append((i , latency))
        memory_list.append((i , abs(memory)))
    
    fig1 , ax1 = plt.subplots()
    fig2 , ax2 = plt.subplots()
    fig3 , ax3 = plt.subplots()
    

    ax1.plot(*zip(*split_list) , linestyle='dotted',alpha=0.8)
    ax1.set_xticks(np.arange(4, 19))
    ax1.set_title("Split Point Layer vs Server Cores")
    ax1.set_xlabel("Server Cores")
    ax1.set_ylabel("Split Layer")    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax2.plot(*zip(*latency_list) , linestyle='dotted',alpha=0.8)
    ax2.set_xticks(np.arange(4,19))
    ax2.set_title("Latency at split layer vs Server Cores")
    ax2.set_xlabel("Server Cores")
    ax2.set_ylabel("Latency")    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax3.plot(*zip(*memory_list) ,linestyle='dotted' ,alpha=0.8)
    ax3.set_xticks(np.arange(4,19))
    ax3.set_title("Memory at split layer vs Server Cores")
    ax3.set_xlabel("Server Cores")
    ax3.set_ylabel("Memory Consumed")    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


    plt.show()
    
    fig1.savefig('split-point-server-cores.png')
    fig2.savefig('Latency-server-cores.png')
    fig3.savefig('Memory-server-cores.png')

#TO PLOT GRAPHS AFTER CHANGING NUMBER OF LAYERS   
plot_changing_layers(settings)
#TO PLOT GRAPHS AFTER CHANGING THE KERNEL SIZE 
plot_changing_kernel_size(settings)
#TO PLOT AFTER CHANGING THE BANDWIDTH
plot_changing_bandwidth(settings)
#TO PLOT AFTER CHANGING TOTAL AVAILABLE MEMORY 
plot_changing_edge_memory(settings)
#TO PLOT AFTER CHANGING TOTAL EDGE CPU COMPUTATION SPEED
plot_changing_edge_computation(settings)
#TO PLOT AFTER CHANGING SERVER COMPUTATION SPEED
plot_changing_server_computation(settings)
#TO PLOT AFTER CHANGING EDGE CORES
plot_changing_edge_cores(settings)
#TO PLOT AFTER CHANGING SERVER CORES
plot_changing_server_cores(settings)