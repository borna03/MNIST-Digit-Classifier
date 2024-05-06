import pygame

pygame.init()
font_size = 32
Button_Background = (94, 142, 193)
font = pygame.font.Font('freesansbold.ttf', font_size)
aa = (231 , 230 , 230)
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)


class Button:
    def __init__(self, x, y, width, height , bkColor , color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = ''
        self.bkColor = bkColor
        self.color = color

    def draw(self, window):
        tekstdemo = pygame.font.Font('freesansbold.ttf', 35)
        text = tekstdemo.render(self.text, True, self.color, self.bkColor)
        textRect = text.get_rect()
        textRect.center = (self.x + (self.width/2), self.y + ((self.height-font_size) / 2 + 20))

        pygame.draw.rect(window, self.bkColor, pygame.Rect(self.x, self.y, self.width, self.height), 0, 7)
        window.blit(text, textRect)
