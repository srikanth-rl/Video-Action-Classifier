# Test the model..
test_frames = preprocess_video(
    test_video_path, desired_height, desired_width, num_frames)
test_input = preprocess_input(np.array(test_frames))

increased_size_frames = [cv2.resize(
    frame, (2 * desired_width, 2 * desired_height)) for frame in test_frames[:10]]

predictions = model.predict(test_input)
predicted_class = np.argmax(predictions[0])

fig, axs = plt.subplots(1, 10, figsize=(20, 3))

for i in range(10):
    axs[i].imshow(cv2.cvtColor(increased_size_frames[i], cv2.COLOR_BGR2RGB))
    axs[i].axis('off')
    axs[i].set_title(f"Frame {i + 1}")

fig.suptitle(f"Predicted Action: {class_names[predicted_class]}", fontsize=18)
plt.show()
