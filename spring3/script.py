import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pygame
import sys
import math

# Fysiske parametere
m1 = 1.0  # kg
m2 = 2.0  # kg
k = 200.0  # N/m
L0 = 5.0  # m

# Startbetingelser
initial_state = [
    0, 0,  # x1, y1
    0, 0,  # vx1, vy1
    10, -10,  # x2, y2
    10, 20  # vx2, vy2
]

# Tidsintervall
t_end = 10.0
num_points = 1000
t = np.linspace(0, t_end, num_points)


def system_equations(state, t):
    """
    Differensialligninger for to-partikkel fjærsystem
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

    # Beregn avstanden mellom partiklene
    dx = x2 - x1
    dy = y2 - y1
    current_length = np.sqrt(dx ** 2 + dy ** 2)

    # Unngå null-divisjon
    if current_length == 0:
        return [vx1, vy1, 0, 0, vx2, vy2, 0, 0]

    # Fjærkraftkomponenter
    force_magnitude = k * (current_length - L0)
    fx = force_magnitude * (dx / current_length)
    fy = force_magnitude * (dy / current_length)

    # Newtons 2. lov: akselerasjoner
    ax1 = fx / m1
    ay1 = fy / m1
    ax2 = -fx / m2
    ay2 = -fy / m2

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]


# Løs differensialligningene
solution = odeint(system_equations, initial_state, t)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("To-Partikkel Fjærsystem")

# Skalering
scaling_factor = 20.0
center_x = WIDTH / 2
center_y = HEIGHT / 2

# Knapp-innstillinger
button_width = 100
button_height = 50
button_margin = 20

button_start = pygame.Rect(20, HEIGHT - 70, button_width, button_height)
button_reset = pygame.Rect(140, HEIGHT - 70, button_width, button_height)
button_pause = pygame.Rect(260, HEIGHT - 70, button_width, button_height)

# Font for tekst
font = pygame.font.SysFont("Arial", 24)


def to_screen(x, y):
    return (center_x + x * scaling_factor, center_y - y * scaling_factor)


def draw_buttons(screen, simulation_running, paused):
    # Start/Fortsett knapp
    button_color = (128, 128, 128) if simulation_running and not paused else (0, 255, 0)
    pygame.draw.rect(screen, button_color, button_start)

    # Reset knapp
    pygame.draw.rect(screen, (255, 0, 0), button_reset)

    # Pause knapp
    button_color = (0, 255, 255) if paused else (128, 128, 128)
    pygame.draw.rect(screen, button_color, button_pause)

    # Tekst på knappene
    font = pygame.font.SysFont('Arial', 24)
    start_text = "Start" if not simulation_running else "Fortsett"
    screen.blit(font.render(start_text, True, (0, 0, 0)), (button_start.x + 20, button_start.y + 10))
    screen.blit(font.render('Reset', True, (0, 0, 0)), (button_reset.x + 20, button_reset.y + 10))
    screen.blit(font.render('Pause', True, (0, 0, 0)), (button_pause.x + 20, button_pause.y + 10))


def draw_particles(x1, y1, x2, y2):
    pygame.draw.circle(screen, (255, 0, 0), to_screen(x1, y1), 10)
    pygame.draw.circle(screen, (0, 0, 255), to_screen(x2, y2), 10)


def draw_spring(x1, y1, x2, y2):
    pygame.draw.line(screen, (255, 255, 255), to_screen(x1, y1), to_screen(x2, y2), 2)


def draw_values(x1, y1, vx1, vy1, ax1, ay1, x2, y2, vx2, vy2, ax2, ay2, t):
    """
    Tegner numeriske verdier som posisjon, hastighet og akselerasjon.
    """
    values = [
        f"Tid: {t:.2f} s",
        f"Partikkel 1: P=({x1:.2f}, {y1:.2f}), V=({vx1:.2f}, {vy1:.2f}), A=({ax1:.2f}, {ay1:.2f})",
        f"Partikkel 2: P=({x2:.2f}, {y2:.2f}), V=({vx2:.2f}, {vy2:.2f}), A=({ax2:.2f}, {ay2:.2f})"
    ]
    y_offset = 20
    for text in values:
        rendered_text = font.render(text, True, (255, 255, 255))
        screen.blit(rendered_text, (20, y_offset))
        y_offset += 30


# Animasjonsløkke
clock = pygame.time.Clock()
running = True
simulation_running = False
paused = False
index = 0
max_index = len(t) - 1

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos

            # Start/Fortsett knapp
            if button_start.collidepoint(mouse_pos):
                if not simulation_running or paused:
                    simulation_running = True
                    paused = False

            # Reset knapp
            elif button_reset.collidepoint(mouse_pos):
                simulation_running = False
                paused = False
                index = 0

            # Pause knapp
            elif button_pause.collidepoint(mouse_pos) and simulation_running:
                paused = not paused

    screen.fill((0, 0, 0))

    # Hent posisjoner og hastigheter
    x1, y1, vx1, vy1 = solution[index, 0], solution[index, 1], solution[index, 2], solution[index, 3]
    x2, y2, vx2, vy2 = solution[index, 4], solution[index, 5], solution[index, 6], solution[index, 7]

    # Beregn akselerasjoner
    state = solution[index]
    derivatives = system_equations(state, t[index])
    ax1, ay1 = derivatives[2], derivatives[3]
    ax2, ay2 = derivatives[6], derivatives[7]

    # Tegn fjær, partikler og verdier
    draw_spring(x1, y1, x2, y2)
    draw_particles(x1, y1, x2, y2)
    draw_values(x1, y1, vx1, vy1, ax1, ay1, x2, y2, vx2, vy2, ax2, ay2, t[index])

    # Tegn knappene
    draw_buttons(screen, simulation_running, paused)

    pygame.display.flip()
    clock.tick(60)

    if simulation_running and not paused:
        index += 1
        if index > max_index:
            simulation_running = False
            index = max_index

pygame.quit()

# Plot resultatene
plt.figure(figsize=(15, 10))

# Posisjoner
plt.subplot(2, 2, 1)
plt.title('Partikkel 1 - Posisjon')
plt.plot(t, solution[:, 0], label='X1')
plt.plot(t, solution[:, 1], label='Y1')
plt.xlabel('Tid (s)')
plt.ylabel('Posisjon (m)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.title('Partikkel 2 - Posisjon')
plt.plot(t, solution[:, 4], label='X2')
plt.plot(t, solution[:, 5], label='Y2')
plt.xlabel('Tid (s)')
plt.ylabel('Posisjon (m)')
plt.legend()
plt.grid(True)

# Hastigheter
plt.subplot(2, 2, 3)
plt.title('Hastighet')
plt.plot(t, solution[:, 2], label='V1x')
plt.plot(t, solution[:, 3], label='V1y')
plt.plot(t, solution[:, 6], label='V2x')
plt.plot(t, solution[:, 7], label='V2y')
plt.xlabel('Tid (s)')
plt.ylabel('Hastighet (m/s)')
plt.legend()
plt.grid(True)

# Bane
plt.subplot(2, 2, 4)
plt.title('Partiklenes Bane')
plt.plot(solution[:, 0], solution[:, 1], label='Partikkel 1')
plt.plot(solution[:, 4], solution[:, 5], label='Partikkel 2')
plt.xlabel('X-posisjon (m)')
plt.ylabel('Y-posisjon (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()