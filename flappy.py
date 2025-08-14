#I developed the Flappy Bird game from the ground up, and implemented both the Neural Network (NN) and Genetic Algorithm (GA) logic to automate gameplay. 
import pygame
import random
import neat
import os
import time
import numpy as np
from neat.genome import DefaultGenome
from neat.reproduction import DefaultReproduction
import matplotlib.pyplot as plt
import pandas as pd  


# Global counter
generation = 0

pygame.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)


# Constants
WINDOW_WIDTH = 600  
WINDOW_HEIGHT = 800
FLOOR = 730
PIPE_GAP = 200
BIRD_START_X = 230
BIRD_START_Y = 350
best_score = 0
best_scores = []


generations_list = []
scores_list = []
best_scores_list = []

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Initialize display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Flappy Bird NEAT')
clock = pygame.time.Clock()

# Global variables
global SPEED_MULTIPLIER
SPEED_MULTIPLIER = 30
generation = 0

def display_info(screen, score, gen, alive_count):
    """Display score, generation, best score and alive agents info"""
    global best_score
    best_score = max(score, best_score)
    
    # Score
    score_label = STAT_FONT.render(f"Score: {score}", True, WHITE)
    screen.blit(score_label, (10, 10))
    
    # Generation
    gen_label = STAT_FONT.render(f"Gen: {gen}", True, WHITE)
    screen.blit(gen_label, (10, 50))
    
    # Best Score
    best_label = STAT_FONT.render(f"Best: {best_score}", True, WHITE)
    screen.blit(best_label, (10, 90))
    
    # Alive Agents
    alive_label = STAT_FONT.render(f"Alive: {alive_count}", True, WHITE)
    screen.blit(alive_label, (10, 130))

def plot_best_scores(best_scores):
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot graph 
    ax1.plot(best_scores)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Score')
    ax1.set_title('Best Score per Generation')
    
    # Create table data
    data = {
        'Generation': generations_list,
        'Score': scores_list,
        'Best Score': best_scores_list
    }
    df = pd.DataFrame(data)
    
    # Display table on right subplot
    ax2.axis('off')
    table = ax2.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.2, 0.2, 0.6, 0.6])  
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax2.set_title('Game Statistics')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

class Bird:
    def __init__(self):
        self.x = BIRD_START_X
        self.y = BIRD_START_Y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.radius = 20  

    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def update(self):
        self.tick_count += 1
        # Physics 
        displacement = self.velocity * self.tick_count + 0.5 * 3 * self.tick_count**2
        
        # Terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16
        if displacement < 0:
            displacement -= 2
            
        self.y = self.y + displacement

    def draw(self, screen):
        """Draw bird as a circle"""
        pygame.draw.circle(screen, WHITE, 
                         (int(self.x), int(self.y)), 
                         self.radius)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap_y = random.randint(150, WINDOW_HEIGHT - 150)
        self.width = 50
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randint(50, 450)
        self.top = self.height - 300
        self.bottom = self.height + PIPE_GAP

    def update(self):
        self.x -= 5  # Pipe speed

    def draw(self, screen):
        # Draw top pipe
        pygame.draw.rect(screen, GREEN, 
                        (self.x, 0, self.width, self.gap_y - PIPE_GAP//2))
        # Draw bottom pipe
        pygame.draw.rect(screen, GREEN, 
                        (self.x, self.gap_y + PIPE_GAP//2, self.width, WINDOW_HEIGHT))

    def collide(self, bird):
        """Check if bird collides with pipe"""
        bird_mask = pygame.Rect(bird.x - bird.radius, bird.y - bird.radius, 
                              bird.radius * 2, bird.radius * 2)
        
        top_pipe = pygame.Rect(self.x, 0, 
                             self.width, self.gap_y - PIPE_GAP//2)
        bottom_pipe = pygame.Rect(self.x, self.gap_y + PIPE_GAP//2,
                                self.width, WINDOW_HEIGHT)
        
        return bird_mask.colliderect(top_pipe) or bird_mask.colliderect(bottom_pipe)

class Genome(DefaultGenome):
    def mutate(self, config):
        # Weight mutation
        r = random.random()
        if r < config.weight_mutate_rate:
            for conn in self.connections.values():
                if random.random() < 0.5:
                    conn.weight += random.gauss(0, config.weight_mutate_power)
                    conn.weight = max(min(conn.weight, config.weight_max_value), config.weight_min_value)
                else:
                    conn.weight = random.uniform(config.weight_min_value, config.weight_max_value)

        # Node addition mutation
        r = random.random()
        if r < config.node_add_prob:
            conn_to_split = random.choice(list(self.connections.values()))
            ng = self.create_node(config, conn_to_split)
            self.nodes[ng.key] = ng

        # Connection addition mutation
        r = random.random()
        if r < config.conn_add_prob:
            self.add_connection_mutation(config)

def two_point_crossover(genome1, genome2, config):
    child = DefaultGenome(0)
    child.connections = {}
    child.nodes = {}

    # Two-point crossover for connections
    points = sorted(random.sample(range(len(genome1.connections)), 2))
    for i, key in enumerate(genome1.connections.keys()):
        if i < points[0] or i > points[1]:
            child.connections[key] = genome1.connections[key].copy()
        else:
            if key in genome2.connections:
                child.connections[key] = genome2.connections[key].copy()

    # Two-point crossover for nodes
    points = sorted(random.sample(range(len(genome1.nodes)), 2))
    for i, key in enumerate(genome1.nodes.keys()):
        if i < points[0] or i > points[1]:
            child.nodes[key] = genome1.nodes[key].copy()
        else:
            if key in genome2.nodes:
                child.nodes[key] = genome2.nodes[key].copy()

    return child

def tournament_select(members, tournament_size=3):
    """Tournament selection for parent selection"""
    # Select random tournament participants 
    tournament = random.sample(members, min(tournament_size, len(members)))
    # Return the one with highest fitness
    return max(tournament, key=lambda x: x[1].fitness)[1]

class Reproduction(DefaultReproduction):
    def reproduce(self, config, species, pop_size, generation):
        new_population = {}
        
        # Sort species by fitness
        species_data = []
        for sid, s in species.species.items():
            if len(s.members) > 0:
                species_data.append((sid, s))
        species_data.sort(key=lambda x: x[1].fitness)
        
        # Generate new population
        for sid, s in species_data:
            # Get members
            members = list(s.members.items())
            
            # keep best performers
            if len(members) > config.species_elitism:
                members.sort(key=lambda x: x[1].fitness, reverse=True)
                for i in range(config.species_elitism):
                    new_population[members[i][0]] = members[i][1]
            
            # Tournament selection for crossover
            while len(new_population) < pop_size:
                # Select parents using tournament selection
                parent1 = tournament_select(members)
                parent2 = tournament_select(members)
                
                # Create and mutate child
                child = two_point_crossover(parent1, parent2, config)
                child.mutate(config)
                new_population[len(new_population)] = child
        
        return new_population

# Update run function to use custom classes
def eval_genomes(genomes, config):
    global generation, best_scores, best_score
    global generations_list, scores_list, best_scores_list
    generation += 1
    
    nets = []
    birds = []
    ge = []
    pipes = [Pipe(WINDOW_WIDTH)]
    score = 0
    
    # Initialize populations
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome.fitness = 0
        ge.append(genome)

    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].width:
                pipe_ind = 1

        # Store birds to remove
        birds_to_remove = []
        
        # Update birds
        for x, bird in enumerate(birds):
            if x >= len(ge) or x >= len(nets):  
                continue
                
            bird.update()
            ge[x].fitness += 0.1
            
            # Neural network
            output = nets[x].activate((bird.y / WINDOW_HEIGHT, (pipes[pipe_ind].gap_y - bird.y) / WINDOW_HEIGHT, (pipes[pipe_ind].x - bird.x) / WINDOW_WIDTH))

            if output[0] > 0.5:
                bird.jump()

            # Check boundaries
            if bird.y + bird.radius > WINDOW_HEIGHT or bird.y < 0:
                birds_to_remove.append(x)

        # Remove birds safely
        for index in sorted(birds_to_remove, reverse=True):
            if index < len(birds):
                birds.pop(index)
            if index < len(nets):
                nets.pop(index)
            if index < len(ge):
                ge[index].fitness -= 2
                ge.pop(index)

        # Update pipes
        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.update()
            
            # Check collisions
            for x, bird in enumerate(birds[:]):
                if pipe.collide(bird) and x < len(ge):
                    ge[x].fitness -= 2
                    if x < len(birds):
                        birds.pop(x)
                    if x < len(nets):
                        nets.pop(x)
                    if x < len(ge):
                        ge.pop(x)
                    continue

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.width < 0:
                rem.append(pipe)

        # Update score and pipes
        if add_pipe:
            score += 1
            # Set fitness equal to score
            for genome in ge:
                genome.fitness = score
            pipes.append(Pipe(WINDOW_WIDTH))

        for r in rem:
            pipes.remove(r)

        # Draw
        screen.fill(BLACK)
        for pipe in pipes:
            pipe.draw(screen)
        for bird in birds:
            bird.draw(screen)
        
        display_info(screen, score, generation, len(birds))
        pygame.display.flip()
        
        if score >= 50:
            # Store data for table
            generations_list.append(generation)
            scores_list.append(score)
            best_scores_list.append(best_score)
            run = False
            break
    
    # Store data for table at the end of each generation
    generations_list.append(generation)
    scores_list.append(score)
    best_scores_list.append(best_score)
    
    best_scores.append(best_score)
    if score >= 50:
        plot_best_scores(best_scores)
        pygame.quit()
        quit()

def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    winner = pop.run(eval_genomes, 100) 
    plot_best_scores(best_scores)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)