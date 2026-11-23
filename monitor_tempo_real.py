import os

# Acelera a abertura da webcam no Windows (deve ficar antes do import cv2)
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
import platform
import threading
import urllib.request

IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import winsound

# Índices anatômicos do modelo BlazePose (33 pontos)
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOME_JANELA = "Monitor de Postura em Tempo Real - Unochapeco"


def garantir_modelo_pose(caminho_modelo="pose_landmarker_lite.task"):
    """Baixa o modelo pré-treinado do MediaPipe caso ainda não exista na pasta."""
    if os.path.exists(caminho_modelo):
        return caminho_modelo

    urls = [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ]
    print("Baixando modelo pose_landmarker_lite.task (apenas na primeira vez)...", flush=True)
    for url in urls:
        try:
            urllib.request.urlretrieve(url, caminho_modelo)
            return caminho_modelo
        except Exception:
            pass
    raise RuntimeError("Não foi possível baixar o arquivo 'pose_landmarker_lite.task'.")


def emitir_alerta_sonoro():
    """Toca um bipe curto em segundo plano para não travar a imagem da câmera."""
    def _tocar():
        if IS_WINDOWS:
            winsound.Beep(1200, 180)
        else:
            print("\a", end="", flush=True)

    threading.Thread(target=_tocar, daemon=True).start()


def calcular_angulo_ombros(ombro_esq, ombro_dir):
    """Calcula a inclinação lateral (em graus) da linha dos ombros."""
    dx = ombro_dir[0] - ombro_esq[0]
    dy = ombro_dir[1] - ombro_esq[1]
    if dx == 0:
        return 90.0
    angulo = abs(math.degrees(math.atan2(dy, dx)))
    return 180.0 - angulo if angulo > 90.0 else angulo


def abrir_fluxo_camera(indice, usar_dshow=False):
    """Abre a webcam na thread principal usando MSMF (padrão Windows) ou DSHOW."""
    if not IS_WINDOWS:
        return cv2.VideoCapture(indice), "AUTO"
    if usar_dshow:
        return cv2.VideoCapture(indice, cv2.CAP_DSHOW), "DSHOW"
    return cv2.VideoCapture(indice, cv2.CAP_MSMF), "MSMF"


def main():
    caminho_modelo = garantir_modelo_pose()

    base_options = python.BaseOptions(model_asset_path=caminho_modelo)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    indice_cam = 0
    usar_dshow = False
    cap, nome_driver = abrir_fluxo_camera(indice_cam, usar_dshow)

    # Parâmetros ergonômicos
    razao_pescoco_base = 0.62       # Referência da altura cabeça-ombros
    tolerancia_curvatura = 0.82     # Queda > 18% indica que o usuário curvou para frente
    limite_inclinacao_ombro = 10.0  # Inclinação lateral máxima entre os ombros (graus)
    tempo_limite_alerta = 2.5       # Segundos em má postura para ficar VERMELHO e apitar

    inicio_ma_postura = None
    ultimo_bipe = 0.0
    total_frames_analisados = 0
    frames_postura_correta = 0
    frames_ma_postura = 0
    mensagem_toast = ""
    tempo_toast = 0.0
    ultimo_timestamp_ms = 0
    falhas_aquecimento = 0

    os.makedirs("evidencias", exist_ok=True)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    print("Monitor em tempo real iniciado! Pressione [Q] na janela da camera para sair.", flush=True)

    while True:
        ret, frame = cap.read() if (cap is not None and cap.isOpened()) else (False, None)

        # Aguarda o aquecimento inicial do sensor (LED da webcam) sem fechar o programa
        if not ret or frame is None:
            falhas_aquecimento += 1
            tela_espera = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)
            cv2.putText(
                tela_espera, f"INICIANDO CAMERA {indice_cam} ({nome_driver})...",
                (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                tela_espera, "Se demorar, aperte [T] para trocar camera ou [B] para driver",
                (55, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA
            )
            cv2.imshow(NOME_JANELA, tela_espera)
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("t"):
                indice_cam = (indice_cam + 1) % 3
                cap.release()
                cap, nome_driver = abrir_fluxo_camera(indice_cam, usar_dshow)
                falhas_aquecimento = 0
            elif tecla == ord("b"):
                usar_dshow = not usar_dshow
                cap.release()
                cap, nome_driver = abrir_fluxo_camera(indice_cam, usar_dshow)
                falhas_aquecimento = 0
            continue

        falhas_aquecimento = 0
        frame = cv2.flip(frame, 1)
        altura, largura, _ = frame.shape

        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= ultimo_timestamp_ms:
            timestamp_ms = ultimo_timestamp_ms + 1
        ultimo_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        resultados = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_texto = "AGUARDANDO PESSOA NA CAMERA..."
        cor_status = (150, 150, 150)  # Cinza
        alerta_vermelho_ativo = False
        razao_pescoco_atual = 0.0
        angulo_ombro_atual = 0.0

        if resultados.pose_landmarks:
            landmarks = resultados.pose_landmarks[0]

            orelha_esq = (landmarks[LEFT_EAR].x * largura, landmarks[LEFT_EAR].y * altura)
            orelha_dir = (landmarks[RIGHT_EAR].x * largura, landmarks[RIGHT_EAR].y * altura)
            ombro_esq = (landmarks[LEFT_SHOULDER].x * largura, landmarks[LEFT_SHOULDER].y * altura)
            ombro_dir = (landmarks[RIGHT_SHOULDER].x * largura, landmarks[RIGHT_SHOULDER].y * altura)

            ponto_medio_orelhas = (
                int((orelha_esq[0] + orelha_dir[0]) * 0.5),
                int((orelha_esq[1] + orelha_dir[1]) * 0.5),
            )
            ponto_medio_ombros = (
                int((ombro_esq[0] + ombro_dir[0]) * 0.5),
                int((ombro_esq[1] + ombro_dir[1]) * 0.5),
            )

            largura_ombros = math.hypot(
                ombro_dir[0] - ombro_esq[0],
                ombro_dir[1] - ombro_esq[1],
            )

            if largura_ombros > 20.0:
                distancia_vertical = ponto_medio_ombros[1] - ponto_medio_orelhas[1]
                razao_pescoco_atual = distancia_vertical / largura_ombros
                angulo_ombro_atual = calcular_angulo_ombros(ombro_esq, ombro_dir)

                cabeca_projetada = razao_pescoco_atual < (razao_pescoco_base * tolerancia_curvatura)
                ombros_desnivelados = angulo_ombro_atual > limite_inclinacao_ombro

                total_frames_analisados += 1
                agora = time.time()

                if cabeca_projetada or ombros_desnivelados:
                    frames_ma_postura += 1
                    if inicio_ma_postura is None:
                        inicio_ma_postura = agora

                    tempo_em_alerta = agora - inicio_ma_postura

                    if tempo_em_alerta >= tempo_limite_alerta:
                        alerta_vermelho_ativo = True
                        status_texto = "ALERTA: CORRIJA SUA POSTURA!"
                        cor_status = (0, 0, 255)  # VERMELHO

                        if agora - ultimo_bipe > 1.5:
                            emitir_alerta_sonoro()
                            ultimo_bipe = agora
                    else:
                        status_texto = f"ATENCAO: POSTURA INCLINADA ({tempo_em_alerta:.1f}s)"
                        cor_status = (0, 215, 255)  # AMARELO
                else:
                    frames_postura_correta += 1
                    inicio_ma_postura = None
                    status_texto = "POSTURA CORRETA"
                    cor_status = (0, 200, 0)  # VERDE

                # Desenha as linhas de alinhamento no corpo
                p_ombro_esq = (int(ombro_esq[0]), int(ombro_esq[1]))
                p_ombro_dir = (int(ombro_dir[0]), int(ombro_dir[1]))

                cv2.line(frame, p_ombro_esq, p_ombro_dir, cor_status, 4)
                cv2.line(frame, ponto_medio_ombros, ponto_medio_orelhas, cor_status, 4)

                cv2.circle(frame, p_ombro_esq, 8, (255, 255, 255), -1)
                cv2.circle(frame, p_ombro_dir, 8, (255, 255, 255), -1)
                cv2.circle(frame, ponto_medio_orelhas, 8, cor_status, -1)
                cv2.circle(frame, ponto_medio_ombros, 8, cor_status, -1)

        # Efeito visual vermelho na tela inteira quando o alerta dispara
        if alerta_vermelho_ativo:
            pulso = 0.18 + 0.10 * math.sin(time.time() * 10.0)
            camada_vermelha = np.full_like(frame, (0, 0, 255))
            cv2.addWeighted(camada_vermelha, pulso, frame, 1.0 - pulso, 0, frame)

        # Borda colorida ao redor de toda a janela (Verde / Amarelo / Vermelho)
        espessura_borda = 14 if alerta_vermelho_ativo else 6
        cv2.rectangle(frame, (0, 0), (largura - 1, altura - 1), cor_status, espessura_borda)

        # Barra superior de status
        cv2.rectangle(frame, (0, 0), (largura, 58), cor_status, -1)
        cv2.putText(
            frame, status_texto, (18, 39),
            cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Painel discreto de métricas
        cv2.rectangle(frame, (15, 70), (350, 165), (25, 25, 25), -1)
        cv2.putText(
            frame, f"Indice Vertical: {razao_pescoco_atual:.2f} (Base: {razao_pescoco_base:.2f})",
            (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Inclinacao Ombros: {angulo_ombro_atual:.1f} graus",
            (25, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Camera: {indice_cam} ({nome_driver})",
            (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA
        )

        # Rodapé de comandos
        cv2.rectangle(frame, (0, altura - 32), (largura, altura), (20, 20, 20), -1)
        cv2.putText(
            frame, "[C] Calibrar Postura   [S] Salvar Print   [T] Trocar Cam   [Q] Sair",
            (15, altura - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA
        )

        if mensagem_toast and (time.time() - tempo_toast < 2.5):
            cv2.putText(
                frame, mensagem_toast, (20, altura - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA
            )

        cv2.imshow(NOME_JANELA, frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("c"):
            if razao_pescoco_atual > 0:
                razao_pescoco_base = razao_pescoco_atual
                inicio_ma_postura = None
                mensagem_toast = f"Postura calibrada! Base: {razao_pescoco_base:.2f}"
                tempo_toast = time.time()
        elif tecla == ord("s"):
            nome_arquivo = f"evidencias/evidencia_{int(time.time())}.png"
            cv2.imwrite(nome_arquivo, frame)
            mensagem_toast = f"Print salvo: {nome_arquivo}"
            tempo_toast = time.time()
        elif tecla == ord("t"):
            indice_cam = (indice_cam + 1) % 3
            cap.release()
            cap, nome_driver = abrir_fluxo_camera(indice_cam, usar_dshow)
            mensagem_toast = f"Trocado para Camera {indice_cam} ({nome_driver})"
            tempo_toast = time.time()
        elif tecla == ord("b"):
            usar_dshow = not usar_dshow
            cap.release()
            cap, nome_driver = abrir_fluxo_camera(indice_cam, usar_dshow)
            mensagem_toast = f"Driver alterado para {nome_driver}"
            tempo_toast = time.time()

    if total_frames_analisados > 0:
        pct_ok = (frames_postura_correta / total_frames_analisados) * 100
        pct_bad = (frames_ma_postura / total_frames_analisados) * 100
        print("\n--- RESUMO DA SESSAO ---")
        print(f"Frames analisados: {total_frames_analisados}")
        print(f"Postura correta:   {frames_postura_correta} ({pct_ok:.1f}%)")
        print(f"Ma postura:        {frames_ma_postura} ({pct_bad:.1f}%)")

    landmarker.close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()