


# https://github.com/SamarpanCoder2002/Graph-Traversing-Visualizer
# https://www.kaggle.com/code/kmader/visualizing-graph-traversal

                                      # Graph Traversing #
from tkinter import *
import time

class GraphTraversal:
    def __init__(self, root):
        self.window = root
        self.make_canvas = Canvas(self.window,bg="chocolate",relief=RAISED,bd=7,width=1000,height=1000)
        self.make_canvas.pack()

        # Status label initialization
        self.status = None

        # Some list initialization bt default
        self.vertex_store = []
        self.total_circle = []
        self.queue_bfs = []
        self.stack_dfs = []

        # Some default function call
        self.basic_set_up()
        self.make_vertex()

    def basic_set_up(self):
        heading = Label(self.make_canvas,text="Graph Traversing Visualization",bg="chocolate",fg="yellow",font=("Arial",20,"bold","italic"))
        heading.place(x=50,y=10)

        bfs_btn = Button(self.window,text="BFS",font=("Arial",15,"bold"),bg="black",fg="green",relief=RAISED,bd=8,command=self.bfs_traversing)
        bfs_btn.place(x=20,y=530)

        dfs_btn = Button(self.window, text="DFS", font=("Arial", 15, "bold"), bg="black", fg="green", relief=RAISED, bd=8, command=self.dfs_traversing)
        dfs_btn.place(x=400, y=530)

        self.status = Label(self.make_canvas,text="Not Visited",bg="chocolate",fg="brown",font=("Arial",20,"bold","italic"))
        self.status.place(x=50,y=450)

    def make_vertex(self):# Vertex with connection make
        for i in range(3**10):
            self.total_circle.append(i)
            

        self.total_circle[0] = self.make_canvas.create_oval(80,420,110,450,width=3)
        self.make_canvas.create_text(80,420,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[1] = self.make_canvas.create_oval(160, 280, 190, 310, width=3)
        self.make_canvas.create_text(160,280,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[2] = self.make_canvas.create_oval(160, 420, 190, 450, width=3)
        self.make_canvas.create_text(160,420,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[3] = self.make_canvas.create_oval(160, 560, 190, 590, width=3)
        self.make_canvas.create_text(160,560,fill="green",font="Times 20 italic bold",text="sell")
#-----------------------------------------------------------------------------------------------------------------------------------------

        self.total_circle[4] = self.make_canvas.create_oval(230, 150, 260, 180, width=3)
        self.make_canvas.create_text(230,150,fill="green",font="Times 20 italic bold",text="sell")
        self.total_circle[5] = self.make_canvas.create_oval(230, 200, 260, 230, width=3)
        self.make_canvas.create_text(230,200,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[6] = self.make_canvas.create_oval(230, 300, 260, 330, width=3)
        self.make_canvas.create_text(230,300,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[7] = self.make_canvas.create_oval(230, 400, 260, 430, width=3)
        self.make_canvas.create_text(230,400,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[8] = self.make_canvas.create_oval(230, 450, 260, 480, width=3)
        self.make_canvas.create_text(230,450,fill="green",font="Times 20 italic bold",text="sell")
        self.total_circle[9] = self.make_canvas.create_oval(230, 600, 260, 630, width=3)
        self.make_canvas.create_text(230,600,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[10] = self.make_canvas.create_oval(230, 650, 260, 680, width=3)
        self.make_canvas.create_text(230,650,fill="darkblue",font="Times 20 italic bold",text="hold")

        #-----------------------------------------------------------------------------------------------------------------------------------------

        self.total_circle[11] = self.make_canvas.create_oval(500, 20, 530, 50, width=3)
        self.make_canvas.create_text(500,20,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[12] = self.make_canvas.create_oval(500, 70, 530, 100, width=3)
        self.make_canvas.create_text(500, 70,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[13] = self.make_canvas.create_oval(500, 120, 530, 150, width=3)
        self.make_canvas.create_text(500,120,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[14] = self.make_canvas.create_oval(500, 160, 530, 190, width=3)
        self.make_canvas.create_text(500,160,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[15] = self.make_canvas.create_oval(500, 200, 530, 230, width=3)
        self.make_canvas.create_text(500,200,fill="green",font="Times 20 italic bold",text="sell")
        self.total_circle[16] = self.make_canvas.create_oval(500, 250, 530, 280, width=3)
        self.make_canvas.create_text(500,250,fill="green",font="Times 20 italic bold",text="sell")
        self.total_circle[17] = self.make_canvas.create_oval(500, 300, 530, 330, width=3)
        self.make_canvas.create_text(500,300,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[18] = self.make_canvas.create_oval(500, 340, 530, 370, width=3)
        self.make_canvas.create_text(500,340,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[19] = self.make_canvas.create_oval(500, 380, 530, 410, width=3)
        self.make_canvas.create_text(500,380,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[20] = self.make_canvas.create_oval(500, 420, 530, 450, width=3)
        self.make_canvas.create_text(500,420,fill="green",font="Times 20 italic bold",text="sell")
        self.total_circle[21] = self.make_canvas.create_oval(500, 460, 530, 490, width=3)
        self.make_canvas.create_text(500,460,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[22] = self.make_canvas.create_oval(500, 500, 530, 530, width=3)
        self.make_canvas.create_text(500,500,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[23] = self.make_canvas.create_oval(500, 540, 530, 570, width=3)
        self.make_canvas.create_text(500,540,fill="green",font="Times 20 italic bold",text="sell")
        self.total_circle[24] = self.make_canvas.create_oval(500, 580, 530, 610, width=3)
        self.make_canvas.create_text(500,580,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[25] = self.make_canvas.create_oval(500, 620, 530, 650, width=3)
        self.make_canvas.create_text(500,620,fill="red",font="Times 20 italic bold",text="buy")
        self.total_circle[26] = self.make_canvas.create_oval(500, 660, 530, 690, width=3)
        self.make_canvas.create_text(500,660,fill="darkblue",font="Times 20 italic bold",text="hold")
        self.total_circle[27] = self.make_canvas.create_oval(500, 700, 530, 730, width=3)
        self.make_canvas.create_text(500,700,fill="green",font="Times 20 italic bold",text="sell")

        self.make_connector_up(0, 1)
        self.make_connector_down(0, 2)
        self.make_connector_down(0, 3)
        self.collector_connector3(0,1,2,3)

        self.make_connector_up(1, 4)
        self.make_connector_down(1, 5)
        self.collector_connector(1, 4, 5)

        self.make_connector_up(2, 6)
        self.make_connector_down(2, 7)
        self.make_connector_down(2, 8)
        self.collector_connector3(2, 6, 7, 8)

        self.make_connector_up(3, 9)
        self.make_connector_down(3, 10)
        self.collector_connector(3, 9, 10)

        self.make_connector_down(4, 11)
        self.make_connector_down(4, 12)
        self.collector_connector(4, 11, 12)

        self.make_connector_up(5, 13)
        self.make_connector_down(5, 14)
        self.make_connector_down(5, 15)
        self.collector_connector3(5, 13, 14, 15)

        self.make_connector_up(6, 16)
        self.make_connector_down(6, 17)
        self.collector_connector(6, 16, 17)

        self.make_connector_up(7, 18)
        self.make_connector_down(7, 19)
        self.make_connector_down(7, 20)
        self.collector_connector3(7, 18, 19, 20)

        self.make_connector_up(8, 21)
        self.make_connector_down(8, 22)
        self.collector_connector(8, 21, 22)

        self.make_connector_up(9, 23)
        self.make_connector_down(9, 24)
        self.collector_connector(9, 23, 24)

        self.make_connector_up(10, 25)
        self.make_connector_down(10, 26)
        self.make_connector_down(10, 27)
        self.collector_connector3(10, 25, 26, 27)


        print(self.vertex_store)

    def make_connector_up(self,index1,index2):# Up node connection make
        first_coord = self.make_canvas.coords(self.total_circle[index1])# Source node coordinates
        second_coord = self.make_canvas.coords(self.total_circle[index2])# Destination node coordinates
        line_start_x = (first_coord[0]+first_coord[2]) / 2# Connector line start_x
        line_end_x = (second_coord[0]+second_coord[2]) / 2# Connector line end_x
        line_start_y = (first_coord[1]+first_coord[3]) / 2# Connector line start_y
        line_end_y = (second_coord[1]+second_coord[3]) / 2# Connector line end_y
        self.make_canvas.create_line(line_start_x+10,line_start_y-10,line_end_x-10,line_end_y+10,width=3)

    def make_connector_down(self,index1,index2):# Down node connection make
        first_coord = self.make_canvas.coords(self.total_circle[index1])# Source node coordinates
        second_coord = self.make_canvas.coords(self.total_circle[index2])# Destination node coordinates
        line_start_x = (first_coord[0] + first_coord[2]) / 2# Connector line start_x
        line_end_x = (second_coord[0] + second_coord[2]) / 2# Connector line end_x
        line_start_y = (first_coord[1] + first_coord[3]) / 2# Connector line start_y
        line_end_y = (second_coord[1] + second_coord[3]) / 2# Connector line end_y
        self.make_canvas.create_line(line_start_x+12 , line_start_y +5, line_end_x - 12, line_end_y -5, width=3)

    def collector_connector(self,source,connector1,connector2):# All about node data collect and store
        temp = []
        temp.append(self.total_circle[source])

        if connector1:
            temp.append(self.total_circle[connector1])
        else:
            temp.append(None)

        if connector2:
            temp.append(self.total_circle[connector2])
        else:
            temp.append(None)

        self.vertex_store.append(temp)

    def collector_connector3(self,source,connector1,connector2, connector3):# All about node data collect and store
        temp = []
        temp.append(self.total_circle[source])

        if connector1:
            temp.append(self.total_circle[connector1])
        else:
            temp.append(None)

        if connector2:
            temp.append(self.total_circle[connector2])
        else:
            temp.append(None)

        if connector3:
            temp.append(self.total_circle[connector3])
        else:
            temp.append(None)

        self.vertex_store.append(temp)

    def binary_search(self,start,end,find_it_as_source):# Binary search algorithm use here
        while start<=end:
            mid = int((start+end)/2)
            if self.vertex_store[mid][0] == find_it_as_source:
                return self.vertex_store[mid]
            elif self.vertex_store[mid][0] < find_it_as_source:
                start = mid + 1
            else:
                end = mid - 1
            #mid+=1
        return self.vertex_store[mid]
        #return -1

    def bfs_traversing(self):
        try:
            self.status['text'] = "Red: Visited"
            self.queue_bfs.append(self.vertex_store[0][0])
            count = 0
            #while self.queue_bfs:
            for i in range(58):
                temp = self.binary_search(0,9,self.queue_bfs[0])
                if temp != -1:
                   if temp[1]:
                      self.queue_bfs.append(temp[1])
                   if temp[2]:
                      self.queue_bfs.append(temp[2])
                take_vertex = self.queue_bfs.pop(0)
                #take_vertex = [0,1,2,3,4,5,6]
                print(take_vertex)
                #self.make_canvas.itemconfig(take_vertex,fill="red")
                self.make_canvas.itemconfig(count,fill="black")
                self.window.update()
                self.window.update()
                time.sleep(0.3)
                count+=1
            self.status['text'] = "All node Visited"
        except:
            print("Force stop error")

    def dfs_traversing(self):
        try:
            self.status['text'] = "Blue: Visited"
            self.stack_dfs.append(self.vertex_store[0][0])
            while self.stack_dfs:
                take_vertex = self.stack_dfs.pop()
                print(take_vertex)
                self.make_canvas.itemconfig(take_vertex, fill="blue")
                self.window.update()
                time.sleep(0.3)
                temp = self.binary_search(0, 9, take_vertex)
                if temp != -1:
                   if temp[1]:
                      self.stack_dfs.append(temp[1])
                   if temp[2]:
                      self.stack_dfs.append(temp[2])
            self.status['text'] = "All node Visited"
        except:
            print("Force stop error")

if __name__ == '__main__':
    window = Tk()
    window.title("Graph Traversal Visualizer")
    window.geometry("400x600")
    window.maxsize(500,600)
    window.minsize(500,600)
    window.config(bg="orange")
    GraphTraversal(window)
    window.mainloop()