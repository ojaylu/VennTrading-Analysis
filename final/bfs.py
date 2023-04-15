from heapq import heappush, heappop
import collections
from visualization import GraphTraversal
from tkinter import *
import time
#predictions = [1,2,3,4,5,6,7,8,9,10]
predictions = [0.4, 0.7, 0,9, 0.8, 0.3, 0.56, 0.7, 0.8, 0.23, 0.19, 0.78, 0.65, 0.87, 0.42, 0.49, 0.1, 0.3, 0.5, 0.7]
def bfs(startState, predictions):
    window = Tk()
    window.title("Graph Traversal Visualizer")
    window.geometry("400x600")
    window.maxsize(1000,800)
    window.minsize(1000,800)
    window.config(bg="orange")
    #GraphTraversal(window)
    window.mainloop()
    frontier = collections.deque([(0, startState)])
    print('Initial frontier:',list(frontier))
    count = 1
    while frontier:
        i = predictions[count]
        node = frontier.popleft()
        count = len(node[1])
        #print('count = ', count)
        stateSpaceGraph={'b':[(i,'s'),(0,'h')],'h':[(-i,'b'),(i,'s'), (0,'h')],'s':[(-i,'b'),(0,'h')]}
        for j in range(1, len(node[1])):
            if node[1][-j] == 'h':
                continue
            elif node[1][-j] == 's':
                stateSpaceGraph={'b':[(i,'s'),(0,'h')],'h':[(-i,'b'), (0,'h')],'s':[(-i,'b'),(0,'h')]}
                break
            elif node[1][-j] == 'b':
                stateSpaceGraph={'b':[(i,'s'),(0,'h')],'h':[(i,'s'), (0,'h')],'s':[(-i,'b'),(0,'h')]}
                break
        if count >= len(predictions)-1: 
            final = []
            for i in max(frontier)[1]:
                if i == 'b':
                    final.append(1)
                elif i =='s':
                    final.append(-1)
                elif i =='h':
                    final.append(0)
            return max(frontier), final
        #print('Exploring:',node[1][-1],'...')
        for child in stateSpaceGraph[node[1][-1]]:
            frontier.append((node[0]+child[0],node[1]+child[1]))
        #input()
        #print('the frontier now is', list(frontier))


#print(bfs('b', predictions))
