import cv2
import numpy as np
import glob
import os


def blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calibrar_camara(ruta_imagenes, patron=(9, 6), tam_cuadrado=25.0, min_blur=30):

    imagenes = glob.glob(os.path.join(ruta_imagenes, "*.jpg"))

    if not imagenes:
        raise FileNotFoundError("No se encontraron imágenes en la ruta especificada.")

    objp = np.zeros((patron[0] * patron[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patron[0], 0:patron[1]].T.reshape(-1, 2) * float(tam_cuadrado)

    objpoints = []
    imgpoints = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    flags_classic = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    usando_sb = hasattr(cv2, "findChessboardCornersSB")

    tam_imagen = None
    omitidas_borrosas = 0
    fallos_lectura = 0
    fallos_tablero = 0

    for fname in imagenes:
        img = cv2.imread(fname)

        if img is None:
            print(f"[WARN] No se pudo leer: {fname}")
            fallos_lectura += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if tam_imagen is None:
            tam_imagen = gray.shape[::-1]

        s = blur_score(gray)

        if s < min_blur:
            print(f"[INFO] Muy borrosa, se omite: {fname} (score={s:.1f})")
            omitidas_borrosas += 1
            continue

        ret, corners = False, None

        if usando_sb:
            ret, corners = cv2.findChessboardCornersSB(
                gray, patron, flags=cv2.CALIB_CB_ACCURACY
            )

        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, patron, flags_classic)
            if ret:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Tablero NO detectado en: {fname}")
            fallos_tablero += 1

    print("\n--- Resumen de Procesamiento ---")
    print(f"Total imágenes: {len(imagenes)}")
    print(f"Válidas para calibración: {len(objpoints)}")
    print(f"Omitidas (borrosas): {omitidas_borrosas}")
    print(f"Fallos (lectura/detección): {fallos_lectura + fallos_tablero}")

    if len(objpoints) < 10:
        print("[WARN] Se recomienda usar al menos 10-15 imágenes.")

    if not objpoints:
        raise RuntimeError("No hay puntos suficientes para calibrar.")

    # Calibración
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, tam_imagen, None, None
    )

    errs = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error_img = np.mean(np.linalg.norm(imgpoints[i] - proj, axis=2))
        errs.append(float(error_img))

    errs = np.array(errs)

    print("\n=== Resultados ===")
    print(f"RMS Global: {rms:.4f}")
    print(f"Error medio: {errs.mean():.4f} px")

    # -----------------------
    # Construir JSON serializable
    # -----------------------

    resultado = {
        "rms": float(rms),

        "intrinsic_matrix": mtx.tolist(),

        "distortion_coefficients": dist.flatten().tolist(),

        "pattern": {
            "columns": patron[0],
            "rows": patron[1],
            "square_size_mm": float(tam_cuadrado)
        },

        "image_size": {
            "width": int(tam_imagen[0]),
            "height": int(tam_imagen[1])
        },

        "reprojection_error": {
            "mean": float(errs.mean()),
            "max": float(errs.max()),
            "per_image": errs.tolist()
        },

        "statistics": {
            "total_images": len(imagenes),
            "valid_images": len(objpoints),
            "blur_skipped": omitidas_borrosas,
            "read_failures": fallos_lectura,
            "detection_failures": fallos_tablero
        }
    }

    return resultado

if __name__ == "__main__":
    from pathlib import Path
    import json

    patron = (9, 6)          # (cols, rows)
    tam_cuadrado = 25.0      # mm
    ruta_carpetas = "./tablero_x_mm"
    min_blur = 30
    save_dir = "./undistorted_values_canon_18-55"
    
    directorio = Path(ruta_carpetas)
    carpetas = [P for P in directorio.iterdir() if P.is_dir()]
    for carpeta in carpetas:
        print(f"\n=== Procesando carpeta: {carpeta.name} ===")
        resultado = calibrar_camara(
            ruta_imagenes=str(carpeta),
            patron=patron,
            tam_cuadrado=tam_cuadrado,
            min_blur=min_blur
        )
        with open(Path(save_dir) / f"calib_{carpeta.name}.json", "w") as f:
            json.dump(resultado, f, indent=4)


