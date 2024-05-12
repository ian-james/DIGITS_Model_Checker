# Open a CV2 window and display the image
# Have a main function that is modular and the image is static

import cv2

def main():
    # Load the image
    image = cv2.imread("./media/Images/hands_PNG944.png")
    if(image is None):
        print("Failed to load image.")
        return

    # Display the image
    cv2.imshow("Image", image)

    # Wait for a key press
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
