import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pygame
from Button import Button
import random  # Import random library at the beginning of your file

bkColor = (21, 21, 21)

screenW, screenH = 1200, 800
screen = pygame.display.set_mode((screenW, screenH))
pygame.display.set_caption('Snake game')
screen.fill(bkColor)
pygame.display.flip()

Gap = 20

np.set_printoptions(linewidth=200)

marginX = 100
marginY = 100

grid = np.zeros((28, 28))
tempGrid = np.zeros((28, 28))

StartButton = Button(700, 100, 200, 85, (94, 142, 193), (231, 230, 230))
StartButton.text = 'Evaluate'

EvaluatedNumber = Button(700, 200, 280, 55, (231, 230, 230), (94, 142, 193))
EvaluatedNumber.text = 'null'


def draw(screen, grid):
    screen.fill(bkColor)
    StartButton.draw(screen)
    EvaluatedNumber.draw(screen)
    row, col = 28, 28
    for i in range(row):
        for j in range(col):
            if grid[i][j] == 0:
                pygame.draw.rect(screen, (255, 255, 255), (marginX + i * Gap, marginY + j * Gap, Gap, Gap))
            else:
                pygame.draw.rect(screen, (0, 0, 0), (marginX + i * Gap, marginY + j * Gap, Gap, Gap), 1)
            pygame.draw.rect(screen, (0, 0, 0), (marginX + i * Gap, marginY + j * Gap, Gap, Gap), 1)
    pygame.display.update()


def returnPos(x, y, gap, margX, margY):
    x = (x - margX) // gap
    y = (y - margY) // gap

    return y, x


model = tf.keras.models.load_model('handwritten.keras')
#
# img = cv2.imread(f'testpicture.png')[:, :, 0]
# img = np.invert(np.array([img]))

def normalize_input(grid):
    # Normalize the grid to be between 0 and 1
    normalized_grid = np.array(grid) / 255.0
    return normalized_grid  # Reshape for the model if needed


run = True
a = [[], [], []]

while run:
    draw(screen, tempGrid)

    mouse = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if StartButton.x <= mouse[0] <= StartButton.x + StartButton.width and StartButton.y <= mouse[1] <= StartButton.y + StartButton.height:
                arr = [[]]
                for item in grid:
                    arr[0].append(item)
                arr = np.array(arr)
                pnormalized_array = normalize_input(arr)
                np.set_printoptions(linewidth=np.inf)
                
                prediction = model.predict(pnormalized_array)
                EvaluatedNumber.text = f'{np.argmax(prediction)}'

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_c:
                grid = np.zeros((28, 28))
                tempGrid = np.zeros((28, 28))

        if pygame.mouse.get_pressed()[0]:
            mousex, mousey = pygame.mouse.get_pos()
            if marginX <= mousex <= (marginX + Gap * 28) and marginY <= mousey <= (marginY + Gap * 28):
                x, y = returnPos(mousex, mousey, Gap, marginX, marginY)
                random_gray_value = random.randint(25, 255)
                
                tempGrid[y][x] = random_gray_value 
                grid[x][y] = random_gray_value

        if event.type == pygame.QUIT:
            pygame.quit()
            run = False

print(grid)
