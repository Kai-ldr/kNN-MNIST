import pygame
import numpy as np
from PIL import Image
from kNN import kNeuralNetwork

data = np.genfromtxt(fname='mnist_train.csv', delimiter=',')
data = np.delete(data, 0, 0)
data = data[0:10000, :]

allLabels = data[:, 0]
data = np.delete(data, 0, 1)

training = data[0:7500, :]
trainingLabels = allLabels[0:7500]

kNN = kNeuralNetwork(10, prints=False)
kNN.fit(training, trainingLabels)

pygame.init()

DRAW_W, DRAW_H = 420, 420
PANEL_W        = 360
HINT_H         = 36
WIN_W          = DRAW_W + PANEL_W
WIN_H          = DRAW_H + HINT_H

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Draw a digit  |  ENTER = predict    C = clear")

canvas = pygame.Surface((DRAW_W, DRAW_H))
canvas.fill((0, 0, 0))

font_title  = pygame.font.SysFont("monospace", 20, bold=True)
font_pred   = pygame.font.SysFont("monospace", 26, bold=True)
font_bar    = pygame.font.SysFont("monospace", 17, bold=True)
font_hint   = pygame.font.SysFont("monospace", 15)

drawing      = False
brush_radius = 9

result = {
    "prediction":   None,
    "likelihoods":  None,
    "closest_surf": None,
}

MATCH_SIZE = 168   # 28x28 scaled to 168x168 (6x zoom)

BG_PANEL   = (18,  18,  30)
COL_BORDER = (60,  60,  90)
COL_TITLE  = (160, 160, 255)
COL_PRED   = ( 80, 220,  80)
COL_BAR_BG = (45,  45,  65)
COL_BAR_FG = ( 80, 160, 255)
COL_TEXT   = (210, 210, 225)
COL_HINT   = ( 90,  90, 120)
COL_WHITE  = (255, 255, 255)


def draw_brush(surface, pos):
    pygame.draw.circle(surface, COL_WHITE, pos, brush_radius)


def preprocess(surface):
    arr  = pygame.surfarray.array3d(surface)
    arr  = np.transpose(arr, (1, 0, 2))
    gray = np.mean(arr, axis=2)
    gray = np.where(gray > 30, 255, 0).astype(np.uint8)

    coords = np.argwhere(gray > 0)
    if coords.size == 0:
        return np.zeros(784, dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    digit  = gray[y0:y1+1, x0:x1+1]

    h, w   = digit.shape
    scale  = 20.0 / max(h, w)
    new_h  = max(1, int(round(h * scale)))
    new_w  = max(1, int(round(w * scale)))

    img = Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    top  = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas28[top:top + new_h, left:left + new_w] = np.array(img)

    return canvas28.flatten().astype(np.float32)


def array_to_surface(flat_arr, size):
    """Flat 784 array -> scaled pygame Surface (nearest-neighbour = crisp pixels)."""
    img_arr = flat_arr.reshape(28, 28).astype(np.uint8)
    rgb     = np.stack([img_arr] * 3, axis=-1)
    small   = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return pygame.transform.scale(small, (size, size))


def weights_to_percentages(weighted_votes):
    total = sum(weighted_votes.values())
    pcts  = {k: (v / total) * 100.0 for k, v in weighted_votes.items()}
    return sorted(pcts.items(), key=lambda x: -x[1])


def render(font, text, color):
    return font.render(text, False, color)


def draw_panel():
    px = DRAW_W
    pygame.draw.rect(screen, BG_PANEL, (px, 0, PANEL_W, WIN_H))
    pygame.draw.line(screen, COL_BORDER, (px, 0), (px, WIN_H), 2)

    cx = px + PANEL_W // 2
    y  = 14

    if result["prediction"] is None:
        hint = render(font_hint, "Press ENTER to predict", COL_HINT)
        screen.blit(hint, hint.get_rect(center=(cx, WIN_H // 2)))
        return

    t = render(font_title, "Closest Match", COL_TITLE)
    screen.blit(t, t.get_rect(center=(cx, y + t.get_height() // 2)))
    y += t.get_height() + 8

    if result["closest_surf"]:
        ir = result["closest_surf"].get_rect(center=(cx, y + MATCH_SIZE // 2))
        pygame.draw.rect(screen, COL_BORDER, ir.inflate(6, 6), 2)
        screen.blit(result["closest_surf"], ir)
        y += MATCH_SIZE + 14

    p = render(font_pred, f"Prediction:  {int(result['prediction'])}", COL_PRED)
    screen.blit(p, p.get_rect(center=(cx, y + p.get_height() // 2)))
    y += p.get_height() + 14

    hdr = render(font_title, "Neighbour Weights", COL_TITLE)
    screen.blit(hdr, hdr.get_rect(center=(cx, y + hdr.get_height() // 2)))
    y += hdr.get_height() + 8

    LABEL_W = 22
    PCT_W   = 62
    BAR_W   = PANEL_W - LABEL_W - PCT_W - 28
    BAR_H   = 20
    ROW_GAP = 7
    bar_x0  = px + 10 + LABEL_W + 4

    for label, pct in weights_to_percentages(result["likelihoods"]):
        fill = int(BAR_W * pct / 100)

        lbl = render(font_bar, str(int(label)), COL_TEXT)
        screen.blit(lbl, (px + 10, y + (BAR_H - lbl.get_height()) // 2))

        pygame.draw.rect(screen, COL_BAR_BG, (bar_x0, y, BAR_W, BAR_H), border_radius=4)
        if fill > 0:
            pygame.draw.rect(screen, COL_BAR_FG, (bar_x0, y, fill, BAR_H), border_radius=4)

        pct_txt = render(font_bar, f"{pct:5.1f}%", COL_TEXT)
        screen.blit(pct_txt, (bar_x0 + BAR_W + 6, y + (BAR_H - pct_txt.get_height()) // 2))

        y += BAR_H + ROW_GAP


def draw_hints():
    pygame.draw.rect(screen, (10, 10, 20), (0, DRAW_H, DRAW_W, HINT_H))
    pairs = [("ENTER", "predict"), ("C", "clear")]
    x = 10
    y = DRAW_H + (HINT_H - font_hint.get_height()) // 2
    for key, action in pairs:
        k = render(font_hint, f"[{key}]", COL_TITLE)
        a = render(font_hint, f" {action}   ", COL_HINT)
        screen.blit(k, (x, y))
        screen.blit(a, (x + k.get_width(), y))
        x += k.get_width() + a.get_width()


running = True
while running:
    screen.blit(canvas, (0, 0))
    draw_hints()
    draw_panel()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.pos[0] < DRAW_W and event.pos[1] < DRAW_H:
                drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                processed = preprocess(canvas)
                preds, closest_flat, likelihoods = kNN.predict(
                    np.array([processed]),
                    returnClosestMatch=True,
                    returnLikelihoods=True,
                )
                result["prediction"]   = preds[0]
                result["likelihoods"]  = likelihoods
                result["closest_surf"] = array_to_surface(closest_flat, MATCH_SIZE)
                print("Prediction:", int(preds[0]))

            elif event.key == pygame.K_c:
                canvas.fill((0, 0, 0))
                result["prediction"]   = None
                result["likelihoods"]  = None
                result["closest_surf"] = None

    if drawing:
        pos = pygame.mouse.get_pos()
        if pos[0] < DRAW_W and pos[1] < DRAW_H:
            draw_brush(canvas, pos)

pygame.quit()