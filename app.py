import os
import cv2
from flask import Flask, render_template, Response, jsonify, request, session, abort, send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename

# If your model/logic lives in YOLO_Video.video_detection, keep it imported.
# It should accept either a file path or a cv2 frame and return an iterable of frames (np.ndarray BGR).
from YOLO_Video import video_detection

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me")
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "static/files")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Toggle webcam in cloud: set ENABLE_WEBCAM=1 only on your local machine.
ENABLE_WEBCAM = os.getenv("ENABLE_WEBCAM", "0") == "1"

# ------------------------------------------------------------------------------
# Forms
# ------------------------------------------------------------------------------
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _encode_jpg(frame_bgr):
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return None
    return buf.tobytes()

def _mjpeg_stream(frame_iterable):
    for frame in frame_iterable:
        blob = _encode_jpg(frame)
        if blob is None:
            break
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + blob + b"\r\n")

def _video_frames_from_path(path_x):
    """Yield YOLO-processed frames from a video file path."""
    for frame in video_detection(path_x):
        yield frame

def _video_frames_from_webcam(index=0):
    """Open a webcam, process frames with YOLO, and yield MJPEG."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # If your video_detection expects a path, adapt it to accept/return frames.
            # Here we assume video_detection can accept a frame or yields frames when given a frame.
            # If not, run your detection inline here.
            for det_frame in video_detection(frame):  # adapt if your function differs
                yield det_frame
    finally:
        cap.release()

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    session.clear()
    # Ensure you have templates/indexproject.html in your repo
    return render_template("indexproject.html")

@app.route("/favicon.ico")
def favicon():
    # Put a real icon at static/favicon.ico to remove 404s in logs
    return send_from_directory("static", "favicon.ico", mimetype="image/x-icon")

@app.route("/FrontPage", methods=["GET", "POST"])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        session["video_path"] = save_path
    # Ensure you have templates/videoprojectnew.html
    return render_template("videoprojectnew.html", form=form)

@app.route("/video")
def video():
    """Stream processed frames from the last uploaded file."""
    path = session.get("video_path")
    if not path or not os.path.exists(path):
        return jsonify(error="No uploaded video found"), 404
    return Response(_mjpeg_stream(_video_frames_from_path(path)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/webcam", methods=["GET"])
def webcam():
    """Landing page for webcam UI, only if enabled."""
    if not ENABLE_WEBCAM:
        abort(404)
    # Ensure you have templates/ui.html that points to /webapp stream
    return render_template("ui.html")

@app.route("/webapp")
def webapp_stream():
    """Stream webcam MJPEG, only if enabled."""
    if not ENABLE_WEBCAM:
        abort(404)

    # Optional: allow ?index=1 to pick a different camera locally
    try:
        cam_index = int(request.args.get("index", "0"))
    except ValueError:
        cam_index = 0

    def gen():
        try:
            for f in _video_frames_from_webcam(cam_index):
                yield f
        except RuntimeError:
            # Fail fast instead of hanging a Gunicorn worker
            pass

    return Response(_mjpeg_stream(gen()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------------------------------------------------------------------------
# Dev entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Bind to platform port if provided
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
