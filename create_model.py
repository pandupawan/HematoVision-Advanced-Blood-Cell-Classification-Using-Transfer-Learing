from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Define model structure
model = Sequential([
    MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax')  # For 4 blood cell classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save model without optimizer to avoid incompatibility
model.save("Blood Cell.h5", include_optimizer=False)

print("Model saved as 'Blood Cell.h5'")
