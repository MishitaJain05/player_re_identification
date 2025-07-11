## Cross-Camera Player Re-Identification — Report

# 1. Methodology and Approach

The goal of this project was to address the cross-camera player re-identification challenge through the utilization of deep learning–based detection and appearance-based similarity calculation.

The pipeline involves the following:

    Detection with YOLOv8: A light-weight and efficient object detection model (YOLOv8) for detecting players in both camera feeds. The detections are pulled for chosen frames of two videos — a Tacticam and a Broadcast view.

    Feature Embeddings with ResNet18: Each player image cropped is input into a ResNet18 CNN to obtain a 512-dimensional feature embedding that describes the appearance of the player.

    Cosine Similarity Matching: Calculated pairwise cosine similarity between embeddings from the two views in order to determine the most probable matches between cameras. The similarity matrix is sorted and thresholded to pick high-confidence identity matches.

    Output Visualization: The matches are output in the console, showing which Tacticam player corresponds to which Broadcast player, with similarity scores.

# 2. Techniques and Results

YOLOv8n was chosen due to its high inference speed and precise detection of players and referees in both views.

ResNet18-based embeddings were applied (without task-specific fine-tuning)

Cosine similarity gave a strong measure for visual appearance comparison across the camera views.

Results
Successful identification of players and referees in both videos.

Reasonable cross-view matches with similarity scores usually ranging from 0.70–0.91, which reflects visual consistency in player appearance due to differences in camera angles.

Readable console outputs

# 3. Problems Faced

Unavailability of pre-trained Re-Identification models: Employed generic ResNet18 embeddings rather than Re-Identification-specialized networks such as OSNet, TransReID, or AGW. This constrained the discriminative ability for fine-grained identity matching.

Camera view variation: The two camera views are quite different in zoom, angle, and lighting, which makes re-identification harder without models that have been fine-tuned.

Evaluation difficulty: No labeled ground truth was available for player IDs, so evaluation was qualitative by visual inspection.

# 4. Next Steps and Remaining Work

If provided with additional time and resources, I could seek to:

Refine a specialized Re-Identification model.

Add output visualization like drawing bounding boxes and matched identities onto the video itself.

Output the results as annotated videos.

Conclusion
This project provides a proof-of-concept for appearance-based visual player re-identification across multiple cameras and similarity matching. The system is modular and highly extendable, with potential for further improvement in embedding quality, temporal tracking, and evaluation metrics.
