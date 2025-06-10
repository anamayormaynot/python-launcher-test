from flask import Flask, render_template, request
import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model & label encoder
model = load_model("models/raga_model_final.h5")
with open("models/label_encoder.pkl", "rb") as f:
    labelencoder = pickle.load(f)

# Function to extract MFCC features (updated to n_mfcc=40)
def features_extractor(filename):
    y, sr = librosa.load(filename, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs.reshape(1, -1)

# Dictionary containing raga details
raga_info = {
    "asavari": {
        "Aaroh": "S R M P d S'",
        "Avaroh": "S' d P M G R S",
        "Pakad": "d P, M P, G R S",
        "Theory": (
            "Raga Asavari is a morning raga from the Asavari Thaat. It evokes a serious, contemplative mood, "
            "often expressing emotions of pathos or devotion. The raga primarily emphasizes the komal Dha and Gandhar notes, "
            "which contribute to its somber nature. Asavari is often employed in slow-paced compositions that allow the emotions "
            "to unfold gradually. It is popular in both classical and semi-classical forms, particularly in khayal and dhrupad styles. "
            "The mood is ideal for expressing deep reflection or inner turmoil. The raga's resemblance to Carnatic raga 'Natabhairavi' "
            "makes it familiar to audiences across regions. Asavari's meditative essence often lends itself well to morning concerts."
        ),
        "YouTube": [
            "https://youtu.be/8y6BAtq8jr4?si=_z-6IRPJqBd8H5EA",
            "https://youtu.be/A3H0gdUs7cQ?si=gx2O0hPhBRLzxQMo"
        ],
        "Carnatic_Raga": "Natabhairavi"
    },
    "bageshri": {
        "Aaroh": "S G m D n S'",
        "Avaroh": "S' n D m G R S",
        "Pakad": "m D n D m G",
        "Theory": (
            "Raga Bageshri is a late-night raga that embodies a deeply romantic and introspective mood. Known for its rich "
            "emotional texture, it evokes feelings of longing and devotion. The raga avoids the Pancham (P) note, relying heavily "
            "on the interplay between Komal Nishad and Shuddha Gandhar. Bageshri is prominent in both khayal and thumri traditions. "
            "Its emotional depth makes it suitable for themes of separation, reunion, or inner reflection. Often performed in a "
            "leisurely tempo, Bageshri demands expressive exploration of melodic phrases. It is closely related to the Carnatic raga "
            "'Shubhapantuvarali', although the tonal structure differs slightly."
        ),
        "YouTube": [
            "https://youtu.be/wWMZGZnSoEc?si=nyV6t-8a36eu8nib",
            "https://youtu.be/uPLSOevIK-4?si=nquS--w5zTd5JA0A"
        ],
        "Carnatic_Raga": "Shubhapantuvarali"
    },
    "bhairavi": {
        "Aaroh": "S r g m P d n S'",
        "Avaroh": "S' n d P m g r S",
        "Pakad": "d n S' d P, m g r S",
        "Theory": (
            "Raga Bhairavi is a versatile raga that is performed in both classical and semi-classical music. Known for its use of all "
            "komal swaras, Bhairavi evokes a devotional and often deeply emotional sentiment. While traditionally a morning raga, it is "
            "frequently presented as the concluding piece in concerts. Bhairavi is widely employed in bhajans, thumris, and ghazals due "
            "to its emotive quality. The raga's flexibility allows for creative improvisation, making it popular among vocalists and "
            "instrumentalists alike. It shares similarities with Carnatic raga 'Hanumatodi', but the melodic treatment differs significantly."
        ),
        "YouTube": [
            "https://youtu.be/tLXNNejKhJs?si=raPfjPrl4SDt3m9v",
            "https://youtu.be/ldSMDT8BMs8?si=a2iX0OWAIsC4l4Cf"
        ],
        "Carnatic_Raga": "Hanumatodi"
    },
    "bhoopali": {
        "Aaroh": "S R G P D S'",
        "Avaroh": "S' D P G R S",
        "Pakad": "G P D, G R S",
        "Theory": (
            "Raga Bhoopali is a pentatonic (audav) raga known for its serene and uplifting mood. It is considered an evening raga that conveys "
            "feelings of peace, contentment, and devotion. The absence of Komal swaras lends Bhoopali a stable and structured form. It is commonly "
            "taught to beginners in Indian classical music due to its simplicity. Bhoopali's scale closely resembles the Western major scale, making it "
            "accessible to diverse musical audiences. The raga is often used in devotional compositions and bhajans, further enhancing its tranquil aura."
        ),
        "YouTube": [
            "https://www.youtube.com/watch?v=5N7RsI9DZtk",
            "https://www.youtube.com/watch?v=WfKpt-_kLdA"
        ],
        "Carnatic_Raga": "Mohanam"
    },
    "vrindavani_sarang": {
        "Aaroh": "S R M P N S'",
        "Avaroh": "S' N P M R S",
        "Pakad": "M P N P M R S",
        "Theory": (
            "Raga Vrindavani Sarang is a midday raga known for its light and cheerful character. Often associated with Krishna Leela, "
            "it evokes feelings of joy and reverence. The raga is characterized by the prominent use of Shuddha Nishad (N) and the absence of Dhaivat. "
            "Vrindavani Sarang is widely employed in light classical music forms such as thumri, dadra, and bhajan. Its melodic simplicity makes it popular "
            "among both vocalists and instrumentalists. The raga's connection to Braj culture adds to its cultural significance."
        ),
        "YouTube": [
            "https://www.youtube.com/watch?v=7kzGbhNpBXM",
            "https://www.youtube.com/watch?v=F_tkYFq-wGg"
        ],
        "Carnatic_Raga": "Madhyamavati"
    },

    "darbari_kanada": {
        "Aaroh": "S R g M P d n S'",
        "Avaroh": "S' n d P M P g M R S",
        "Pakad": "M P g M R, S",
        "Theory": (
            "Raga Darbari Kanada is a late-night raga that belongs to the Kanada family. It is characterized by its slow tempo, "
            "meandering phrases, and deep, serious tone. The raga evokes a sense of grandeur and is often associated with majestic, royal court settings. "
            "The use of Komal Gandhar and Dhaivat with heavy andolan (oscillation) creates a rich and immersive experience. Darbari Kanada is deeply linked "
            "to the Dhrupad tradition but is also popular in khayal and semi-classical styles. Its complex structure demands careful execution, particularly "
            "with the slow unfolding of melodic phrases. The raga is closely related to the Carnatic raga 'Shahana'."
        ),
        "YouTube": [
            "https://www.youtube.com/watch?v=4elr9ZQjPA8",
            "https://www.youtube.com/watch?v=JSRaWqrU9JY"
        ],
        "Carnatic_Raga": "Shahana"
    },
    "yaman": {
        "Aaroh": "N R G M D N S'",
        "Avaroh": "S' N D P M G R S",
        "Pakad": "N R G, R G M D N",
        "Theory": (
            "Raga Yaman is one of the most popular evening ragas in Hindustani classical music. Known for its calm and romantic mood, "
            "it is often the first raga taught to beginners due to its straightforward structure and beautiful melodic phrases. "
            "Yaman is performed with Teevra Madhyam (M#), giving it a distinctively bright and luminous character. The raga's ascending and descending scales "
            "follow strict rules, creating a smooth and fluid melodic flow. Yaman is commonly featured in khayal, thumri, and bhajan performances. "
            "Its Carnatic counterpart is 'Kalyani', which shares a similar scale structure but varies in treatment and ornamentation."
        ),
        "YouTube": [
            "https://www.youtube.com/watch?v=MTf84M05VxU",
            "https://www.youtube.com/watch?v=XHZ8UR7GBw8"
        ],
        "Carnatic_Raga": "Kalyani"
    },
    "malkauns": {
        "Aaroh": "S g M d n S'",
        "Avaroh": "S' n d M g S",
        "Pakad": "g M d n d M g S",
        "Theory": (
            "Raga Malkauns is a deeply meditative and serious raga, often associated with spiritual introspection. It is believed to have originated from ancient Indian "
            "music traditions and is considered one of the oldest ragas. Malkauns is a pentatonic raga (audav) that omits the Rishabh and Pancham notes. "
            "Its characteristic komal Gandhar, Dhaivat, and Nishad notes create a somber, yet powerful atmosphere. The raga is traditionally performed at midnight, "
            "symbolizing inner peace and strength. Malkauns is closely linked to the Carnatic raga 'Hindolam', which follows a similar note pattern but has a distinct presentation style."
        ),
        "YouTube": [
            "https://www.youtube.com/watch?v=4ADFKdIVvD8",
            "https://www.youtube.com/watch?v=2JfSOM6Me1Q"
        ],
        "Carnatic_Raga": "Hindolam"
    }
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_class = None
    raga_details = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Extract features & predict
            mfccs_scaled_features = features_extractor(filepath)
            predictions = model.predict(mfccs_scaled_features)
            predicted_label = np.argmax(predictions, axis=1)
            prediction_class = labelencoder.inverse_transform(predicted_label)[0]

            # Fetch raga details
            raga_details = raga_info.get(prediction_class, None)

    return render_template("index.html", prediction=prediction_class, raga_details=raga_details)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
