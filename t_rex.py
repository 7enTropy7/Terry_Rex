import pygame
import neat
import os
import random

class Dino:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.vel = 5
        self.jump = False
        self.c = 10
        self.dino_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images","dino.png")))
        
    def jmp(self):
        self.jump = True

    def move(self):
        if self.jump:
            if self.c >=-10:
                neg = -1
                if self.c < 0:
                    neg = 1
                self.y += (self.c**2)*0.5*neg
                self.c -= 1
            else:
                self.jump = False
                self.c = 10
        
    def draw(self,win):
        win.blit(self.dino_img,(self.x,self.y))
    
    def get_mask(self):
        return pygame.mask.from_surface(self.dino_img)
    
class Cactii:
    def __init__(self):
        self.x = 700
        self.y = 400
        self.top = 0
        self.passed = False
        #self.cactus_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images","dirt.png")))
        self.cactus_img = pygame.transform.scale(pygame.transform.scale2x(pygame.image.load(os.path.join("images","cactus.png"))), (random.randrange(30,80), 90))
        self.set_height()

    def set_height(self):
        #self.y = random.randrange(50,450)
        self.top = self.y-self.cactus_img.get_height()
        
    def move(self):
        self.x -= random.randrange(10,30)
        
    def draw(self,win):
        win.blit(self.cactus_img,(self.x,self.y))
        #pygame.draw.rect(win,(0,0,0),(self.x,self.y,30,1))
    
    
    def collide(self,dino):
        
        dino_mask = dino.get_mask()
        cactus_mask = pygame.mask.from_surface(self.cactus_img)
        
        offset = (self.x-round(dino.x),self.top-round(dino.y))
        point = dino_mask.overlap(cactus_mask,offset)
        if point:
            return True
        return False
        

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main,100)



def draw_win(win,dino,cactii,score):
    win.fill((50,50,50))
    for cactus in cactii:
        cactus.draw(win)
    dino.draw(win)
    pygame.font.init()
    score_font = pygame.font.SysFont("Forte",50)
    score_text = score_font.render("Score: "+str(score),1,(0,0,0))
    win.blit(score_text,(10,10))
    pygame.display.update()

def main(genomes,config):
    dinos = []
    cactii = [Cactii()]
    pygame.init()
    win = pygame.display.set_mode((640,480))
    pygame.display.set_caption("Terry Rex")
    nets = []
    ge = []
    run = True
    score = 0

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        dinos.append(Dino(100,370))
        g.fitness = 0
        ge.append(g)

    while run:
        pygame.time.delay(20)

        curr_cactus = 0

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    #break
        
        
        if len(dinos) > 0:
            if len(cactii) >1 and dinos[0].x > cactii[0].x + cactii[0].cactus_img.get_width():
                curr_cactus = 1
        else:
            run = False
            break


        for x,dino in enumerate(dinos):
            dino.move()
            ge[x].fitness += 0.1
            
            output = nets[x].activate((dino.x,dino.y,abs(dino.x-cactii[curr_cactus].x)))

            
            if output[0]>0.5:
                dino.x-=dino.vel
            if output[1]>0.5:
                dino.x+=dino.vel
            if output[2]>0.5:
                dino.jmp()
            
        
        add_cactus = False
        del_cactus = []
        
        for cactus in cactii:

            for x,dino in enumerate(dinos):
                if cactus.collide(dino) or dino.x<0 or dino.x>640:
                    ge[x].fitness -= 1
                    dinos.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                if not cactus.passed and cactus.x<dino.x:
                    cactus.passed = True
                    add_cactus = True
            
            if cactus.x+cactus.cactus_img.get_width()<0:
                del_cactus.append(cactus)
            
            cactus.move()


        if add_cactus:
            score += 1
            for g in ge:
                g.fitness += 5
            cactii.append(Cactii())
        for i in del_cactus:
            cactii.remove(i)
    
        draw_win(win,dino,cactii,score)


if __name__=="__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    run(config_path)