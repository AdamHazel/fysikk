# ===========================
# Imports og biblioteker
# ===========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pygame
import sys
import math

# ===========================
# Fysiske parametere
# ===========================
m1 = 1.0  # kg
m2 = 2.0  # kg
k = 200.0  # N/m
L0 = 5.0  # m

# ===========================
# Startbetingelser
# ===========================
initial_state = [
    -10, 0,  # x1, y1
    0, 20,  # vx1, vy1
    10, 0,  # x2, y2
    0, -10  # vx2, vy2
]

# Tidsintervall for simuleringen
t_end = 10.0
num_points = 1000
t = np.linspace(0, t_end, num_points)


# ===========================
# Fysikkmotor
# ===========================
def system_equations(state, t):
    """
    Differensialligninger for to-partikkel fjærsystem
    state: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

    # Beregn nåværende avstand mellom partiklene
    dx = x2 - x1
    dy = y2 - y1
    current_length = np.sqrt(dx ** 2 + dy ** 2)

    # Fjærkraft
    f_spring = k * (current_length - L0)

    # Normaliserte retningsenheter
    if current_length != 0:
        nx = dx / current_length
        ny = dy / current_length
    else:
        nx, ny = 0, 0

    # Fjærkraftkomponenter
    fx1 = f_spring * nx
    fy1 = f_spring * ny
    fx2 = -fx1
    fy2 = -fy1

    # Akselerasjoner
    ax1 = fx1 / m1
    ay1 = fy1 / m1
    ax2 = fx2 / m2
    ay2 = fy2 / m2

    return [vx1, vy1, ax1, ay1,
            vx2, vy2, ax2, ay2]


# Løs differensialligningene
solution = odeint(system_equations, initial_state, t)

# ===========================
# Pygame oppsett
# ===========================
pygame.init()

WIDTH, HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("To-partikkel fjær simulering")

# Skalering mellom meter og piksler
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


# ===========================
# Hjelpefunksjoner
# ===========================
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


def draw_spring(x1, y1, x2, y2):
    num_of_coils = 20
    dx = x2 - x1
    dy = y2 - y1
    total_length = math.sqrt(dx * dx + dy * dy)

    if total_length == 0:
        return

    direction = (dx / total_length, dy / total_length)
    perpendicular = (-direction[1], direction[0])
    seg_length = total_length / num_of_coils
    spring_amp = 0.2 * total_length
    current_pos = (x1, y1)

    for i in range(num_of_coils):
        offset = spring_amp if i % 2 == 0 else -spring_amp
        next_pos = (
            current_pos[0] + direction[0] * seg_length + perpendicular[0] * offset * 0.05,
            current_pos[1] + direction[1] * seg_length + perpendicular[1] * offset * 0.05
        )
        pygame.draw.line(screen, (255, 255, 255), to_screen(*current_pos), to_screen(*next_pos), 2)
        current_pos = next_pos

    pygame.draw.line(screen, (255, 255, 255), to_screen(*current_pos), to_screen(x2, y2), 2)


# ===========================
# Animasjonsløkke
# ===========================
clock = pygame.time.Clock()
running = True
simulation_running = False
paused = False
index = 0
max_index = len(t) - 1
pRadius = 10

font = pygame.font.SysFont('Arial', 24)

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

    # Hent posisjoner fra løsningen
    x1_sol = solution[index, 0]
    y1_sol = solution[index, 1]
    x2_sol = solution[index, 4]
    y2_sol = solution[index, 5]

    # Hent hastigheter
    vx1_sol = solution[index, 2]
    vy1_sol = solution[index, 3]
    vx2_sol = solution[index, 6]
    vy2_sol = solution[index, 7]

    # Beregn akselerasjoner fra systemligningene
    current_state = [x1_sol, y1_sol, vx1_sol, vy1_sol,
                     x2_sol, y2_sol, vx2_sol, vy2_sol]
    derivatives = system_equations(current_state, t[index])
    ax1_sol = derivatives[2]
    ay1_sol = derivatives[3]
    ax2_sol = derivatives[6]
    ay2_sol = derivatives[7]

    draw_spring(x1_sol, y1_sol, x2_sol, y2_sol)

    # Tegn partiklene
    pygame.draw.circle(screen, (255, 0, 0), to_screen(x1_sol, y1_sol), pRadius)
    pygame.draw.circle(screen, (0, 0, 255), to_screen(x2_sol, y2_sol), pRadius)

    # Vis tid
    time_text = font.render(f"Tid: {t[index]:.2f} s", True, (255, 255, 255))
    screen.blit(time_text, (20, 20))

    # Vis informasjon for partikkel 1
    pos_text_p1 = font.render(f"Posisjon (P1): ({x1_sol:.1f}, {y1_sol:.1f}) m", True, (255, 255, 255))
    vel_text_p1 = font.render(f"Hastighet (P1): ({vx1_sol:.1f}, {vy1_sol:.1f}) m/s", True, (255, 255, 255))
    acc_text_p1 = font.render(f"Akselerasjon (P1): ({ax1_sol:.1f}, {ay1_sol:.1f}) m/s²", True, (255, 255, 255))

    # Vis informasjon for partikkel 2
    pos_text_p2 = font.render(f"Posisjon (P2): ({x2_sol:.1f}, {y2_sol:.1f}) m", True, (255, 255, 255))
    vel_text_p2 = font.render(f"Hastighet (P2): ({vx2_sol:.1f}, {vy2_sol:.1f}) m/s", True, (255, 255, 255))
    acc_text_p2 = font.render(f"Akselerasjon (P2): ({ax2_sol:.1f}, {ay2_sol:.1f}) m/s²", True, (255, 255, 255))

    # Tegn tekst på skjermen
    screen.blit(pos_text_p1, (20, 60))
    screen.blit(vel_text_p1, (20, 90))
    screen.blit(acc_text_p1, (20, 120))

    screen.blit(pos_text_p2, (20, 160))
    screen.blit(vel_text_p2, (20, 190))
    screen.blit(acc_text_p2, (20, 220))

    # Tegn knappene
    draw_buttons(screen, simulation_running, paused)

    pygame.display.flip()
    clock.tick(60)

    # Oppdater indeks hvis simuleringen kjører og ikke er pauset
    if simulation_running and not paused:
        index += 1
        if index > max_index:
            simulation_running = False
            index = max_index

pygame.quit()

# ===========================
# Plotting av resultater
# ===========================
plt.figure(figsize=(15, 10))

# Posisjon
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

# Hastighet
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