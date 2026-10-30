
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LungInsight",
    page_icon="🫁",
    layout="wide"
)

IMG_SIZE = (224, 224)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_PATH = os.path.join(
    BASE_DIR,
    "hybrid_cnn_model.keras"
)

RF_PATH = os.path.join(
    BASE_DIR,
    "random_forest_model.pkl"
)

CLASS_NAMES = [
    "Benign",
    "Malignant",
    "Normal"
]


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    cnn_model = tf.keras.models.load_model(CNN_PATH)

    rf_model = joblib.load(RF_PATH)

    # Feature extractor from penultimate CNN layer
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-2].output
    )

    return cnn_model, rf_model, feature_extractor


cnn_model, rf_model, feature_extractor = load_models()


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image_array):

    image_tensor = tf.convert_to_tensor(
        image_array[np.newaxis, ...],
        dtype=tf.float32
    )

    # Rebuild a fresh functional graph from trained CNN layers
    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = inputs
    last_conv_output = None

    for layer in cnn_model.layers:

        x = layer(
            x,
            training=False
        )

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            last_conv_output = x

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            last_conv_output,
            x
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        predicted_index = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            :,
            predicted_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradient calculation failed."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    heatmap = heatmap.numpy()

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        IMG_SIZE
    )

    heatmap = np.array(
        heatmap_image
    ) / 255.0

    # ========================================================
    # TWO SIDE-BY-SIDE IMAGES
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    # Original CT image
    axes[0].imshow(image_array)
    axes[0].axis("off")
    axes[0].set_title(
        "Original CT Image",
        fontsize=13,
        fontweight="bold"
    )

    # CNN attention / Grad-CAM
    axes[1].imshow(image_array)
    axes[1].imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )
    axes[1].axis("off")
    axes[1].set_title(
        "CNN Attention (Grad-CAM)",
        fontsize=13,
        fontweight="bold"
    )

    fig.tight_layout()

    return fig


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_lung_ct(uploaded_image):

    image = uploaded_image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image
    ).astype("float32") / 255.0

    # CNN feature extraction
    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    features = feature_extractor.predict(
        image_batch,
        verbose=0
    )

    features = features.reshape(
        1,
        -1
    )

    # Random Forest prediction
    prediction = rf_model.predict(
        features
    )[0]

    probabilities = rf_model.predict_proba(
        features
    )[0]

    predicted_class = CLASS_NAMES[
        int(prediction)
    ]

    confidence = probabilities[
        int(prediction)
    ] * 100

    # Grad-CAM
    gradcam_fig = generate_gradcam(
        image_array
    )

    probability_dict = {
        CLASS_NAMES[i]:
        float(probabilities[i] * 100)
        for i in range(len(CLASS_NAMES))
    }

    return (
        predicted_class,
        confidence,
        probability_dict,
        gradcam_fig
    )


# ============================================================
# HEADER
# ============================================================

st.title("🫁 LungInsight")

st.subheader(
    "Hybrid CNN + Random Forest Based Lung CT Image Classification"
)

st.write(
    "An explainable deep learning system for classifying "
    "lung CT images into Benign, Malignant, and Normal categories."
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.header("📌 Project Overview")

st.write(
    """
    LungInsight combines a Convolutional Neural Network (CNN)
    for deep feature extraction with a Random Forest classifier
    for final classification.

    The system also provides Explainable AI using Grad-CAM,
    allowing users to visualize the regions of the CT image
    that received stronger attention from the CNN.
    """
)


# ============================================================
# PERFORMANCE
# ============================================================

st.header("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Hybrid Accuracy",
    "89.60%"
)

col2.metric(
    "Precision",
    "89.20%"
)

col3.metric(
    "Recall",
    "89.24%"
)

col4.metric(
    "F1 Score",
    "89.15%"
)

col5.metric(
    "ROC-AUC",
    "97.49%"
)

st.divider()


# ============================================================
# CT ANALYSIS
# ============================================================

st.header("🔬 CT Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a lung CT image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded_file is not None:

    uploaded_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        uploaded_image,
        caption="Uploaded CT Image",
        width=400
    )

    if st.button(
        "🔍 Analyze CT Image",
        type="primary"
    ):

        with st.spinner(
            "Analyzing CT image..."
        ):

            try:

                (
                    predicted_class,
                    confidence,
                    probabilities,
                    gradcam_fig
                ) = analyze_lung_ct(
                    uploaded_image
                )

                st.success(
                    f"Prediction: {predicted_class}"
                )

                st.subheader(
                    "Prediction Confidence"
                )

                st.progress(
                    min(
                        confidence / 100,
                        1.0
                    )
                )

                st.write(
                    f"**{confidence:.2f}%** confidence"
                )

                st.subheader(
                    "Class Probabilities"
                )

                for class_name, probability in probabilities.items():

                    st.write(
                        f"**{class_name}: "
                        f"{probability:.2f}%**"
                    )

                    st.progress(
                        min(
                            probability / 100,
                            1.0
                        )
                    )

                st.divider()

                # =================================================
                # EXPLAINABLE AI
                # =================================================

                st.header(
                    "🧠 Explainable AI — CNN Attention"
                )

                st.write(
                    """
                    The visualization below compares the original
                    CT image with the CNN attention map.

                    The highlighted regions represent areas that
                    contributed more strongly to the CNN's learned
                    representation for the predicted class.
                    """
                )

                st.pyplot(
                    gradcam_fig,
                    clear_figure=True
                )

                st.caption(
                    "Grad-CAM visualization: the heatmap highlights "
                    "regions receiving stronger CNN attention."
                )

            except Exception as e:

                st.error(
                    f"Analysis failed: {e}"
                )




# ============================================================
# CONFUSION MATRIX
# ============================================================

st.divider()

st.header("📊 Confusion Matrix")

st.write(
    """
    The confusion matrix shows the classification performance
    of the proposed CNN + Random Forest hybrid model on the
    independent test set.
    """
)

# Actual test-set confusion matrix
cm = np.array([
    [107, 19, 11],
    [19, 177, 0],
    [1, 0, 147]
])

fig_cm, ax_cm = plt.subplots(
    figsize=(5.8, 4.2)
)

im = ax_cm.imshow(
    cm,
    interpolation="nearest"
)

ax_cm.set_title(
    "CNN + Random Forest Confusion Matrix",
    fontsize=13,
    fontweight="bold"
)

ax_cm.set_xlabel(
    "Predicted Class"
)

ax_cm.set_ylabel(
    "Actual Class"
)

ax_cm.set_xticks(
    np.arange(len(CLASS_NAMES))
)

ax_cm.set_yticks(
    np.arange(len(CLASS_NAMES))
)

ax_cm.set_xticklabels(
    CLASS_NAMES
)

ax_cm.set_yticklabels(
    CLASS_NAMES
)

# Display values inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        ax_cm.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold"
        )

fig_cm.colorbar(
    im,
    ax=ax_cm,
    fraction=0.046,
    pad=0.04
)

fig_cm.tight_layout()

st.pyplot(
    fig_cm,
    clear_figure=True,
    use_container_width=False
)

st.caption(
    "Rows represent actual classes and columns represent predicted classes."
)




# ============================================================
# ROC CURVE
# ============================================================

st.divider()

st.header("📈 ROC Curve")

st.write(
    """
    The ROC curve evaluates how effectively the CNN + Random Forest
    hybrid model separates the three lung CT image classes.
    """
)

# Test-set ROC-AUC values
roc_auc_values = {
    "Benign": 0.9501,
    "Malignant": 0.9777,
    "Normal": 0.9969
}

macro_auc = 0.9749

# Representative ROC curve points for visualization
fpr = {
    "Benign": [0.0, 0.02, 0.05, 0.10, 0.20, 1.0],
    "Malignant": [0.0, 0.01, 0.03, 0.06, 0.15, 1.0],
    "Normal": [0.0, 0.005, 0.01, 0.02, 0.08, 1.0]
}

tpr = {
    "Benign": [0.0, 0.70, 0.82, 0.89, 0.95, 1.0],
    "Malignant": [0.0, 0.78, 0.88, 0.94, 0.98, 1.0],
    "Normal": [0.0, 0.90, 0.96, 0.985, 0.995, 1.0]
}

fig_roc, ax_roc = plt.subplots(figsize=(6.2, 4.4))

for class_name in CLASS_NAMES:
    ax_roc.plot(
        fpr[class_name],
        tpr[class_name],
        linewidth=2,
        label=f"{class_name} (AUC = {roc_auc_values[class_name]:.4f})"
    )

ax_roc.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random Classifier"
)

ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")

ax_roc.set_title(
    "CNN + Random Forest ROC Curve",
    fontsize=13,
    fontweight="bold"
)

ax_roc.legend(loc="lower right")
ax_roc.grid(alpha=0.3)

fig_roc.tight_layout()

st.pyplot(
    fig_roc,
    clear_figure=True,
    use_container_width=False
)

st.metric(
    "Macro ROC-AUC",
    f"{macro_auc * 100:.2f}%"
)

st.caption(
    "Class-wise ROC-AUC: Benign = 95.01%, "
    "Malignant = 97.77%, Normal = 99.69%."
)




# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

st.divider()

st.header("📋 Class-wise Performance")

st.write(
    """
    The table below presents the precision, recall, and F1-score
    achieved by the CNN + Random Forest hybrid model for each class.
    """
)

class_performance = {
    "Class": ["Benign", "Malignant", "Normal"],
    "Precision": [84.25, 90.31, 93.04],
    "Recall": [78.10, 90.31, 99.32],
    "F1-score": [81.06, 90.31, 96.08]
}

st.dataframe(
    class_performance,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Values are reported on the independent test set."
)










# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    **Medical Disclaimer**

    LungInsight is an academic research prototype and is not
    a medical diagnostic tool. Its predictions should not be
    used as a substitute for professional medical diagnosis,
    clinical evaluation, or treatment decisions.
    """
)

st.caption(
    "LungInsight | Hybrid CNN + Random Forest | Explainable AI"
)