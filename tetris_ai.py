import sys
import time
import pygame
import random
import numpy as np
import math
# ----
width = 600 #Dài của cửa sổ trò chơi
height = 600 #Rộng của cửa sổ trò chơi
gameWidth = 100 #Rộng của scene trò chơi  
gameHeight = 400 #Dài của scene trò chơi
i = 0
j = 0
# ---
class Metrics:
    def __init__(self):
        self.numBlocks = 0
        self.currentHeight = 0 #Tổng chiều cao của các cột (chọn)
        self.columnHeights = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
        self.currentBumpiness = 0 #Độ gập ghềnh của field (chọn)
        self.currentMaxHeight = 0 #Chiều cao cột cao nhất (chọn)
        self.currentNumHoles = 0 #Số lỗ bị chặn trên bởi ít nhất 1 ô đã tô (chọn) 
        self.holesCenterRow = 0 #Tâm của các lỗ theo chiều dọc
        self.holesCenterCol = 0 #Tâm của các lỗ theo chiều ngang
        self.sumHolesRow = 0
        self.sumHolesCol = 0
        self.lines = 0 #Số dòng sắp hoàn thành hoặc có thể hoàn thành (chọn)
        self.nodes = 0 #Số node trên đồ thị cần phải xét

scores = [0, 40, 100, 300, 1200]
# -----
topLeft_x = 2 #bên trái của hoành
topLeft_y = 2
# -------
pygame.font.init()
screen = pygame.display.set_mode((width, height))
# ---------
block_shape = [[[1, 2, 5, 6]],[[1, 5, 9, 13], [4, 5, 6, 7]], [[1, 5, 9, 10], [8, 9, 10, 6], [1, 2, 6, 10], [8, 4, 5, 6]],
               [[2, 6, 10, 9], [4, 5, 6, 10], [2, 1, 5, 9], [4, 8, 9, 10]], [[1, 2, 6, 7], [2, 6, 5, 9]],[[1, 5, 6, 10], [4, 5, 1, 2]],
               [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]]]

block_column = [[[0,1,1,0]],[[0,1,0,0], [1,1,1,1]], [[0,1,1,0], [1,1,1,0], [0,1,1,0], [1,1,1,0]],[[0,1,1,0], [1,1,1,0], [0,1,1,0], [1,1,1,0]],
                [[0,1,1,1], [0,1,1,0]], [[0,1,1,0], [1,1,1,0]], [[1,1,1,0], [1,1,0,0], [1,1,1,0], [0,1,1,0]]]

block_scope_x = [[[1,2]],[[1,1], [0,3]],[[1,2], [0,2], [1,2], [0,2]],[[1,2], [0,2], [1,2], [0,2]],
                [[1,3], [1,2]],[[1,2], [0,2]], [[0,2], [0,1], [0,2], [1,2]]]            

color = ["green", "yellow", "red", "pink", "brown", "purple", "blue"]

class BLOCK:
    x=0
    y=0
    n=0
    def __init__(self, x, y, n) -> None:
        self.x = x
        self.y = y
        self.n = n
        self.style = n%7
        self.rotation = 0
    def image(self):
        return block_shape[self.style][self.rotation] ###cái này để hiển thị ra cái list của block để chiếu trên cái ma trận kia
    def paint(self):
        for m in range (4):
            for n in range(4):
                if m*4+n in self.image():
                    pygame.draw.rect(screen, color[self.style], [(self.x+n)*20+topLeft_x, (self.y+m)*20+topLeft_y, 18, 18], 0)
    def preview(self):
        pygame.draw.rect(screen, "white", [19*20+topLeft_x, 9*20+topLeft_x, 120, 120], 0) 
        pygame.draw.rect(screen, "gray", [19*20+topLeft_x, 9*20+topLeft_x, 120, 120], 2) 
        for m in range (4):
            for n in range(4):
                if m*4+n in block_shape[self.style][0]:
                    pygame.draw.rect(screen, color[self.style], [(n+20)*20+topLeft_x, (m + 10)*20+topLeft_x, 18, 18], 0) 
field = np.empty([31,16])
# field.fill(0)
metrics = Metrics()

currentBlock = BLOCK(6, 0, random.randint(0,100))
nextBlock  = BLOCK(6, 0, random.randint(0,100))

def printScore(score):
    pygame.draw.rect(screen, "white", [19*20+topLeft_x, 5*20+topLeft_x, 120, 60], 0) 
    pygame.draw.rect(screen, "gray", [19*20+topLeft_x, 5*20+topLeft_x, 120, 60], 2) 
    SCORE = SSSCORE.render("Score: " + str(score), True, "black")
    screen.blit(SCORE , (19*20+topLeft_x, 5*20+topLeft_x))

score = 0
SSSCORE = pygame.font.SysFont('Corbel', 20) 
# ------------------------------------------------------------------------------------------------
def update_height_and_holes(line):
    global metrics
    metrics.currentHeight-=15
    metrics.currentMaxHeight-=1
    for i in range (15):
        metrics.columnHeights[i]-=1
    for col in range(15):
        blocked=False
        for row in range (j-1, 29-metrics.columnHeights[col],-1):
            if field[row][col]==1:
                blocked = True
                break
        if blocked == False:
            for row in range(j+1,30,1):
                if field[row][col]==1:
                    break
                else: 
                    field[row][col]=0
                    metrics.currentNumHoles-=1
                    metrics.sumHolesCol-=col
                    metrics.sumHolesRow-=row
      


def break_line():
    zero = 0
    lines = 0
    for y1 in range (29, -1, -1):
        for x1 in range(14, -1, -1):
            if field[y1][x1] < 1:
                zero += 1
        if zero == 0:
            update_height_and_holes(y1)
            lines += 1
            for y2 in range(y1, -1, -1):
                for x2 in range (15, -1, -1):
                    field[y2][x2] = field[y2-1][x2]
        zero = 0
    return lines

def check (block):
    result = True
    for m in range (4):
            for n in range(4):
                if m*4+n in block.image():
                    if m+block.y >= 30 or \
                        n+block.x >= 15 or n+block.x < 0 or \
                        field[m+block.y][n+block.x] > 0:
                        result = False
                       
    return result
def change (current_block, next_block):
    current_block.__init__(6, 0, next_block.n)
    next_block.__init__(6, 0, random.randint(0,100))
def bottom(block):
    global metrics
    metrics.numBlocks+=1
    highest=0
    lowest=3
    for n in range (4): 
        if block_column[block.style][block.rotation][n] == 0:
            continue
        for m in range(3,-1,-1):
            if m*4+n in block.image():
                field[block.y+m][block.x+n] = 1
                highest = block.y+m
        for m in range(4):
            if m*4+n in block.image():
                field[block.y+m][block.x+n] = 1
                lowest = block.y+m
        j=lowest+1
        while(field[j][block.x+n]==0 and j<30):
            field[j][block.x+n]=-1 #ô bị block đặt là -1 trong field
            metrics.sumHolesCol+=block.x+n
            metrics.sumHolesRow+=j
            j+=1
        metrics.columnHeights[block.x+n]+=(j-highest)
        metrics.currentMaxHeight = max(metrics.currentMaxHeight, metrics.columnHeights[block.x+n])
        metrics.currentHeight+=(j-highest)
        metrics.currentNumHoles+=(j-lowest-1)
        
    lines = break_line()
    change(block, nextBlock)
    return lines

def go_down (block):
    block.y += 1 
    if not check(block):
        block.y -= 1
        return bottom(block)
    return 0

def go_side(block, dx):
    block.x += dx
    if not check(block):
        block.x -= dx

def c_rotation (block):
    old_ = block.rotation
    block.rotation = (old_ + 1) % (len(block_shape[block.style]))
    if not check(block):
        block.rotation = old_        
def go_bottom(block):
    while check(block):
        block.y += 1
    block.y -= 1
    return bottom(block)

#------------------------------------------------------------------------------------
font = pygame.font.SysFont('Corbel', 40)  
text1 = font.render('RESTART' , True , "white")
text2 = font.render('QUIT' , True , "white")
text3 = font.render('START' , True , "white")
text4 = font.render('GBFS', True, 'White')
text5 = font.render('GA (SCRATCH)', True, 'White')
text6 = font.render('GA (PRETRAINED)', True, 'White')
# text7 = font.render('POPULATION SIZE', True, 'White')
# text8 = font.render('NUMBER OF GENERATIONS', True, 'White')
h = text1.get_height()
w1 = text1.get_width()
w2 = text2.get_width()
w3 = text3.get_width()
w4 = text4.get_width()
w5 = text5.get_width()
w6 = text6.get_width()
# w7 = text7.get_width()
# w8 = text8.get_width()
# num_generations=0
# population_size=0
done = False


# def restart (mouse):
#     if width/2 - 60 <= mouse[0] <= width/2 - 60 + w1 and height/2-50 <= mouse[1] <= height/2-50+h1:
#         pygame.draw.rect(screen,"gray",[width/2- 60,height/2-50,w1,h1])
#     else:
#         pygame.draw.rect(screen,"black",[width/2- 60,height/2-50,w1,h1])
#     if width/2 - 60 <= mouse[0] <= width/2 - 60 + w2 and height/2 <= mouse[1] <= height/2+h2:
#         pygame.draw.rect(screen,"gray",[width/2- 60,height/2,w2,h2])
#     else:
#         pygame.draw.rect(screen,"black",[width/2- 60,height/2,w2,h2])
#     screen.blit(text1 , (width/2- 60, height/2-50))
#     screen.blit(text2 , (width/2- 60, height/2))
class DropDown():

    def __init__(self, color_menu, color_option, x, y, w, h, font, main, options):
        self.color_menu = color_menu
        self.color_option = color_option
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.main = main
        self.options = options
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1

    def draw(self, surf):
        pygame.draw.rect(surf, self.color_menu[self.menu_active], self.rect, 0)
        msg = self.font.render(self.main, 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center = self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.options):
                rect = self.rect.copy()
                rect.y += (i+1) * self.rect.height
                pygame.draw.rect(surf, self.color_option[1 if i == self.active_option else 0], rect, 0)
                msg = self.font.render(text, 1, (0, 0, 0))
                surf.blit(msg, msg.get_rect(center = rect.center))

    def update(self, event_list):
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)
        
        self.active_option = -1
        for i in range(len(self.options)):
            rect = self.rect.copy()
            rect.y += (i+1) * self.rect.height
            if rect.collidepoint(mpos):
                self.active_option = i
                break

        if not self.menu_active and self.active_option == -1:
            self.draw_menu = False

        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    self.draw_menu = not self.draw_menu
                elif self.draw_menu and self.active_option >= 0:
                    self.draw_menu = False
                    return self.active_option
        return -1

def screen1():
    global done
    global population_size
    done = False
    screen.fill("White")
    mode=-1

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                done = True
                return
            mouse = pygame.mouse.get_pos()
            if width/2 - w3/2 <= mouse[0] <= width/2 + w3/2 and height/2-200 <= mouse[1] <= height/2-200+h:
                pygame.draw.rect(screen,"gray",[width/2-w3/2,height/2-200,w3,h])
            else:
                pygame.draw.rect(screen,"black",[width/2-w3/2,height/2-200,w3,h])
            screen.blit(text3 , (width/2-w3/2,height/2-200,w3,h))
            if width/2 - w4/2 <= mouse[0] <= width/2 + w4/2 and height/2-100 <= mouse[1] <= height/2-100+h:
                pygame.draw.rect(screen,"gray",[width/2-w4/2,height/2-100,w4,h])
            else:
                pygame.draw.rect(screen,"black",[width/2-w4/2,height/2-100,w4,h])
            screen.blit(text4 , (width/2-w4/2,height/2-100,w4,h))
            if width/2 - w5/2 <= mouse[0] <= width/2 + w5/2 and height/2 <= mouse[1] <= height/2+h:
                pygame.draw.rect(screen,"gray",[width/2-w5/2,height/2,w5,h])
            else:
                pygame.draw.rect(screen,"black",[width/2-w5/2,height/2,w5,h])
            screen.blit(text5 , (width/2-w5/2,height/2,w5,h))
            if width/2 - w6/2 <= mouse[0] <= width/2 + w6/2 and height/2+100 <= mouse[1] <= height/2+100+h:
                pygame.draw.rect(screen,"gray",[width/2-w6/2,height/2+100,w6,h])
            else:
                pygame.draw.rect(screen,"black",[width/2-w6/2,height/2+100,w6,h])
            screen.blit(text6 , (width/2-w6/2,height/2+100,w6,h))


            if ev.type == pygame.MOUSEBUTTONDOWN:
                if width/2 - w3/2 <= mouse[0] <= width/2 + w3/2 and height/2-200 <= mouse[1] <= height/2-200+h:
                    mode=0
                elif width/2 - w4/2 <= mouse[0] <= width/2 + w4/2 and height/2-100 <= mouse[1] <= height/2-100+h:
                    mode=1
                elif width/2 - w5/2 <= mouse[0] <= width/2 + w5/2 and height/2 <= mouse[1] <= height/2+h:
                    mode=2
                elif width/2 - w6/2 <= mouse[0] <= width/2 + w6/2 and height/2+100 <= mouse[1] <= height/2+100+h:
                    mode=3
                else:
                    continue
                
                game_screen(mode)
                if done:
                    return
            pygame.display.update()

def game_screen(mode):
    global done
    if mode<2:
        while True:
            if done: 
                return
            start = time.time()
            score = startGame(mode=mode) 
            end = time.time()
            if mode==1:
                print("Time elapsed: {}\nAverage time per block: {}\nNumber of nodes traversed: {}\nScore: {}".format(end-start, (end-start)/metrics.numBlocks, metrics.nodes, score))
            pygame.display.update()      
    else: 
        if mode==2:
            genetic_trainer = GeneticAlgorithm()
        else:
            genetic_trainer = GeneticAlgorithm(pretrained=True)
        genetic_trainer.start_GA()

# def screen2():
#     global done
#     COLOR_INACTIVE = (100, 80, 255)
#     COLOR_ACTIVE = (100, 200, 255)
#     COLOR_LIST_INACTIVE = (255, 100, 100)
#     COLOR_LIST_ACTIVE = (255, 150, 150)
#     list1 = DropDown(
#         [COLOR_INACTIVE, COLOR_ACTIVE],
#         [COLOR_LIST_INACTIVE, COLOR_LIST_ACTIVE],
#         width/2 - w7/2, height/2 - 100, w7, h, 
#         pygame.font.SysFont(None, 30), 
#         text7, [30, 60, 100])
    
#     while True:
#         clock.tick(30)

#         event_list = pygame.event.get()
#         for event in event_list:
#             if event.type == pygame.QUIT:
#                 done = True
#                 return

#         selected_option = list1.update(event_list)
#         if selected_option >= 0:
#             list1.main = list1.options[selected_option]

#         screen.fill("White")
#         list1.draw(screen)
#         pygame.display.flip()
    
# def screen2() :
#     done = False
#     while not done:
#         for ev in pygame.event.get():
#             mouse = pygame.mouse.get_pos()
#             if ev.type == pygame.QUIT:
#                 pygame.quit()
#             else:
#                 restart(mouse)
#                 if ev.type == pygame.MOUSEBUTTONDOWN:
#                     if width/2-60 <= mouse[0] <= width/2 - 60 + w1 and height/2-50 <= mouse[1] <= height/2-50+h1:
#                         startGame()
#                     if width/2-60 <= mouse[0] <= width/2 - 60+ w2 and height/2 <= mouse[1] <= height/2+h2:
#                         pygame.quit()
#         pygame.display.update()
# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
def go_bottom_simulate(block, GA, weights):
    while check(block):
        block.y += 1
    block.y -= 1
    return bottomSimulate(block, GA, weights)

def bottomSimulate(block, GA, weights):
    global metrics
    currentNumHoles = metrics.currentNumHoles
    currentMaxHeight = metrics.currentMaxHeight
    currentHeight = metrics.currentHeight
    columnHeights = metrics.columnHeights.copy()
    # sumHolesRow = metrics.sumHolesRow
    # sumHolesCol = metrics.sumHolesCol


    fieldSimulate = field.copy()
    holesCreated = 0
    holesFreed = 0
    

    #bottom() simulation
    highest=0
    lowest=3
    for n in range (4): 
        if block_column[block.style][block.rotation][n] == 0:
            continue
        for m in range(3,-1,-1):
            if m*4+n in block.image():
                fieldSimulate[block.y+m][block.x+n] = 1
                highest = block.y+m
        for m in range(4):
            if m*4+n in block.image():
                fieldSimulate[block.y+m][block.x+n] = 1
                lowest = block.y+m
        j=lowest+1
        while(fieldSimulate[j][block.x+n]==0 and j<30):
            fieldSimulate[j][block.x+n]=-1 #ô bị block đặt là -1 trong field
            # sumHolesCol2+=block.x+n
            # sumHolesRow2+=j
            j+=1
        columnHeights[block.x+n]+=(j-highest)
        currentMaxHeight = max(currentMaxHeight, columnHeights[block.x+n])
        currentHeight+=(j-highest)
        currentNumHoles+=(j-lowest-1)
        holesCreated+=(j-lowest-1)
        
    #break_line() simulation
    zero = 0
    lines = 0
    for y1 in range (29, -1, -1):
        for x1 in range(14, -1, -1):
            if fieldSimulate[y1][x1] < 1:
                zero += 1
        if zero == 0:
            currentHeight-=15
            currentMaxHeight-=1
            for i in range (15):
                columnHeights[i]-=1
            for col in range(15):
                blocked=False
                for row in range (j-1, 29-columnHeights[col],-1):
                    if fieldSimulate[row][col]==1:
                        blocked = True
                        break
                if blocked == False:
                    for row in range(j+1,30,1):
                        if fieldSimulate[row][col]==1:
                            break
                        else: 
                            fieldSimulate[row][col]=0
                            currentNumHoles-=1
                            holesFreed+=1
                            # sumHolesCol-=col
                            # sumHolesRow-=row

            lines += 1
            for y2 in range(y1, -1, -1):
                for x2 in range (15, -1, -1):
                    fieldSimulate[y2][x2] = fieldSimulate[y2-1][x2]
        zero = 0

    # bumpiness
    bumpiness = 0
    for i in range(14):
        bumpiness+=abs(columnHeights[i]-columnHeights[i+1])

    # heuristic function: lines, holes, bumpiness, max_height
    if GA:
        score = np.matmul([lines, currentNumHoles, bumpiness, currentMaxHeight, currentHeight], weights)
    else: 
        score = (np.exp(lines+1)*(holesFreed+1))/((holesCreated+1)*(currentMaxHeight**2)*(currentHeight**2)*(bumpiness+1))
    return score
    
def best_first_search(CURRENT, GA, weights):
    maxScore = -10000
    style = ()
    for rotation_id in range(len(block_shape[CURRENT.style])):
        left_limit = -block_scope_x[CURRENT.style][rotation_id][0]
        right_limit = 15-block_scope_x[CURRENT.style][rotation_id][1]
        metrics.nodes += right_limit-left_limit
        for x in range(left_limit,right_limit, 1):
            temp = CURRENT
            temp.rotation = rotation_id
            temp.x = x
            temp.y = -1  
            tempScore =  go_bottom_simulate(temp, GA, weights)
            if(maxScore < tempScore):
                maxScore = tempScore
                style = (x, rotation_id)
    return style
# ------------------------------------------------------------------------------------------------
clock = pygame.time.Clock()

def startGame(mode=0, GA=False, weights=[]):
    global currentBlock
    global metrics
    global done
    metrics = Metrics()
    fps = 5
    field.fill(0)   
    done = False
    clock = pygame.time.Clock()
    screen.fill("white")
    score = 0
    while not done:
        nextBlock.preview()
        printScore(score)
        score += scores[go_down(currentBlock)]
        for x1 in range(0, 15, 1):
            for y1 in range (0, 30, 1):
                pygame.draw.rect(screen, "gray", [x1*20+topLeft_x, y1*20+topLeft_y, 18, 18], 0)
        if mode==0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        go_side(currentBlock, -1)
                    if event.key == pygame.K_RIGHT:
                        go_side(currentBlock, 1)
                    if event.key == pygame.K_UP:
                        c_rotation(currentBlock)
                    if event.key == pygame.K_DOWN:
                        score += scores[go_down(currentBlock)]
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True        
            temp = currentBlock
            x, rotation_id = best_first_search(temp, GA, weights)
            currentBlock.x = x
            currentBlock.y = -1
            currentBlock.rotation = rotation_id
            score += scores[go_bottom(currentBlock)]     
        esc = False   
        for a in range(0, 16):
            for b in range (0, 31):
                if field[b][a] > 0:
                    if b==0:
                        esc=True
                    pygame.draw.rect(screen, "black", [a*20+topLeft_x, b*20+topLeft_y, 18, 18], 0)
        currentBlock.paint()
        if esc==True: 
            break
        clock.tick(fps)
        pygame.display.flip()
    return score

class GeneticAlgorithm:
    def __init__(self,population_size=100, num_generations=5, pretrained=False):
        self.population_size = population_size
        self.num_generations = num_generations
        self.last_population = [0 for _ in range(population_size)]
        self.current_population = []
        self.fitness = [0 for _ in range(population_size)]
        self.pretrained = pretrained
        
    def population_randomizer(self):
        self.current_population = [np.random.rand(5)*2-1 for _ in range(self.population_size)]
        for i in range(self.population_size):
            s = np.sqrt(np.sum(np.square(self.current_population[i])))
            self.current_population[i]/=s
            
    def crossover(self, p1, p2):
        if self.fitness[p1]==0 and self.fitness[p2]==0:
            child = (self.last_population[p1]+self.last_population[p2])/2
        else:
            child = self.last_population[p1]*self.fitness[p1]+self.last_population[p2]*self.fitness[p2]
        s = np.sqrt(np.sum(np.square(child)))
        child/=s 
        return child  

    def tournament(self):
        num_chosen = math.floor(self.population_size*0.1)
        iterations = math.floor(self.population_size*0.3)
        concat_length = self.population_size-iterations

        for _ in range(iterations):
            max1 = -10
            max2 = -10
            p1 = -1
            p2 = -1
            indices = np.random.randint(low=0, high=self.population_size, size=num_chosen)
            for j in indices:
                if max1<self.fitness[j]:
                    p1 = j
                    max1 = self.fitness[j]
                elif max2<self.fitness[j]:
                    p2 = j
                    max2 = self.fitness[j]
            self.current_population.append(self.crossover(p1,p2))
        
        t = np.c_[self.fitness, self.last_population]
        a = [x[1:] for x in sorted(t, key=lambda x:x[0], reverse=True)]
        self.current_population += a[:concat_length]

            


    def mutation(self):
        num_mutations = math.floor(self.population_size*0.05) 
        indices = np.random.randint(low=0, high=self.population_size, size=num_mutations)
        for i in indices:
            j = np.random.randint(low=0, high=5)
            self.current_population[i][j]+=(np.random.rand()*0.4-0.2)
            s = np.sqrt(np.sum(np.square(self.current_population[i])))
            self.current_population[i]/=s

    def start_GA(self):
        global done
        mode=2
        if self.pretrained:
            self.current_population = np.load('saved_weights.npy')
            print(self.current_population[0])
            mode=3
        else:
            self.population_randomizer()
        for n in range(self.num_generations):
            print("==========================================")
            for i in range(self.population_size):
                start = time.time()
                self.fitness[i] = startGame(mode=mode, GA=True, weights=self.current_population[i])
                end = time.time()
                if done:
                    return
                pygame.display.update()
                print("generation: {} player: {} fitness: {}".format(n+1,i+1, self.fitness[i]))
                print("Time elapsed: {}\nAverage time per block: {}\nNumber of nodes traversed: {}".format(end-start, (end-start)/metrics.numBlocks, metrics.nodes))
                print(self.current_population[i])
                print("==========================================")
            self.last_population = self.current_population.copy()
            self.current_population = []
            self.tournament()
            self.mutation()
            self.fitness = [0 for _ in range(self.population_size)]
            np.save('saved_weights',self.current_population)
        done = True
    
    def test(self):
        self.population_randomizer()
        np.save("saved_weights", self.current_population)

# m = GeneticAlgorithm()
# m.test()

def run():
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        else:
            screen1()
            if done:
                sys.exit()   


run()