import cv2
import mediapipe as mp
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- SNAP (GRID INVISÍVEL) ---
def snap_para_grid(x, y, tamanho):
    return (
        (x // tamanho) * tamanho + tamanho // 2,
        (y // tamanho) * tamanho + tamanho // 2
    )


class DetectorMaos:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO  # 🔥 ESTABILIZA
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.timestamp = 0

    def detectar(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp += 1

        result = self.detector.detect_for_video(mp_image, self.timestamp)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            h, w, _ = frame.shape

            pontos = []

            for p in hand:
                x, y = int(p.x * w), int(p.y * h)
                pontos.append((x, y))

                # pontos
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # conexões (malha mais leve)
            conexoes = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,13),(13,17),  # base mais estável
                (0,17)
            ]

            for c in conexoes:
                cv2.line(frame, pontos[c[0]], pontos[c[1]], (255, 255, 255), 2)

            return pontos

        return None


def main():
    cap = cv2.VideoCapture(0)
    detector = DetectorMaos()

    cubos = []
    TAM_CUBO = 40

    tempo_pinça = None
    TEMPO_CONSTRUIR = 0.8

    LIMIAR_PINCA = 35
    LIMIAR_MOVER = 45

    movendo = False
    ultimo_ponto = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        pontos = detector.detectar(frame)

        if pontos:
            indicador = pontos[8]
            polegar = pontos[4]
            mindinho = pontos[20]

            dist_pinça = math.hypot(indicador[0] - polegar[0], indicador[1] - polegar[1])
            dist_mover = math.hypot(mindinho[0] - polegar[0], mindinho[1] - polegar[1])

            # --- CONSTRUIR ---
            if dist_pinça < LIMIAR_PINCA:
                if tempo_pinça is None:
                    tempo_pinça = time.time()

                tempo = time.time() - tempo_pinça

                if tempo > TEMPO_CONSTRUIR:
                    x, y = snap_para_grid(indicador[0], indicador[1], TAM_CUBO)

                    if (x, y) not in cubos:
                        cubos.append((x, y))

                    tempo_pinça = None
            else:
                tempo_pinça = None

            # --- MOVER ---
            if dist_mover < LIMIAR_MOVER:
                if not movendo:
                    movendo = True
                    ultimo_ponto = polegar
                else:
                    dx = polegar[0] - ultimo_ponto[0]
                    dy = polegar[1] - ultimo_ponto[1]

                    cubos = [(x + dx, y + dy) for (x, y) in cubos]

                    ultimo_ponto = polegar
            else:
                movendo = False
                ultimo_ponto = None

        # --- DESENHAR CUBOS ---
        for (x, y) in cubos:
            cv2.rectangle(frame,
                          (x - TAM_CUBO // 2, y - TAM_CUBO // 2),
                          (x + TAM_CUBO // 2, y + TAM_CUBO // 2),
                          (255, 0, 0), -1)

        cv2.imshow("Builder Clean", frame)

        tecla = cv2.waitKey(1)

        if tecla == 27:
            break
        elif tecla == ord('c'):
            cubos = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()