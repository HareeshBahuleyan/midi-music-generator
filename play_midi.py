import pygame
from config import config

pygame.init()

# pygame.mixer.init(44100, -16, 2, 2048)
pygame.mixer.music.load(config['generated_dir'] + 'generated.mid')
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.wait(1000)