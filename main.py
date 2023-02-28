import pygame
from pong import Game
import neat
import os
import pickle


class PongGame:
    
    """
    Initializes a new instance of PongGame class with the specified window, width, and height, and sets the game, left_paddle, right_paddle, and ball properties to their respective values in the Game object.
    """
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball
    


    def test_ai(self, genome, config):
        """
        This function tests the performance of an AI player with a given genome and configuration. It creates a feedforward neural network based on the genome, and uses its outputs to control the movement of the right paddle. The function loops until the game is over, and updates the game state and display on each iteration. The game_info variable holds information about the game status, such as the scores and whether the game is over
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            output = net.activate(
                (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        """
        This function trains two genomes (neural networks) against each other using the Pong game. The method creates two networks using the provided genomes and config. Then, it runs a loop where it retrieves the output of the networks based on the current state of the game (position of paddles and ball) and makes decisions on moving the paddles accordingly. The loop runs until one of the players scores a point or hits the ball more than 50 times. At that point, the fitness of each genome is calculated based on the game outcome and the training process ends.
        """
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            output1 = net1.activate(
                (self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decision1 = output1.index(max(output1))

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            output2 = net2.activate(
                (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()

            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
        """
        The calculate_fitness function takes in genome1, genome2, and game_info as parameters. It updates the fitness score of each genome based on the number of hits made by their respective paddles during the game. The fitness score is used to determine the fitness level of each genome and to select the fittest individuals for the next generation during the NEAT training process.
        """
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits


def eval_genomes(genomes, config):
    """
    This function eval_genomes takes in a list of genomes and a configuration object, and evaluates the fitness of each genome by training it against every other genome in the list. It does this by creating a PongGame object for each pair of genomes and calling the train_ai method on it. The fitness of each genome is updated based on the number of hits it made during the game.
    """
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)


def run_neat(config):
    """
    This function runs the NEAT algorithm to evolve neural networks to play Pong. It takes in a configuration object for the NEAT algorithm, creates a new population, adds various reporters to track the progress of the algorithm, and runs the eval_genomes function to evaluate the fitness of each genome in the population. After the algorithm completes, the winning genome is saved to a file using pickle.
    """
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-8')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 20)
    with open("model.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    """
    This function loads the saved winner model from "model.pickle" file and uses it to run the PongGame using the test_ai method of the PongGame class. The config parameter is passed to the test_ai method to create the neural network used in the game.
    """
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    with open("model.pickle", "rb") as f:
        winner = pickle.load(f)

    game = PongGame(window, width, height)
    game.test_ai(winner, config)


if __name__ == "__main__":
    """
    This is the main function that loads the NEAT configuration file, creates a NEAT configuration object, and then either trains the AI using run_neat(config) or tests the AI using test_ai(config).
    """
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # run_neat(config) # Uncomment this line to train the model
    test_ai(config) # Uncomment this line to test the model
