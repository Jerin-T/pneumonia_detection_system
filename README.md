# pneumonia_detection_system
Pneumonia Detection System using X-rays 

This project focused on developing and evaluating deep learning models for the
classification of medical images to detect pneumonia, aiming to enhance diagnostic
accuracy and reliability. The primary objectives were to compare different
convolutional neural network (CNN) architectures and techniques to determine the
most effective model for distinguishing between normal and pneumonia-affected
lungs

#Objectives:
• To develop and evaluate multiple CNN models for pneumonia detection.
• To assess the impact of data augmentation.
• To address issues related to class imbalance and overfitting in medical image
classification.

#Methods Used:
Models Evaluated: VGG19 (initial and improved versions) and InceptionV3
(initial and with data augmentation).
Techniques: Data augmentation and performance evaluation using metrics such as
accuracy, precision, recall, and F1-score.
Test Cases: Included basic functionality, class balance, overfitting checks, edge
cases, and generalization tests.

#Key Findings:
VGG19: The initial version struggled with overfitting and poor generalization,
while the improved version showed better validation performance but still lagged
other models in overall accuracy.

InceptionV3: The initial model had significant difficulties, especially with class
imbalance. However, incorporating data augmentation led to a substantial
improvement, achieving high validation and test accuracies, and effectively
addressing class imbalance issues.

Data Augmentation: Proved to be a crucial technique, enhancing the model’s
ability to generalize and perform well across different classes.Conclusions:
• InceptionV3 with data augmentation emerged as the most effective model,
demonstrating robust performance in classifying both normal and
pneumonia-affected cases.
• The project highlighted the importance of data augmentation and careful
model tuning in improving performance and addressing challenges such as
class imbalance and overfitting.
• Future work should explore additional architectures, advanced data
augmentation techniques, and real-world testing to further enhance model
reliability and applicability in clinical settings.
