import cv2 as cv
import os


def censure(img, x, y, w, h, grid):
    resized = cv.resize(img[y:y+h, x:x+w], (grid, grid), interpolation=cv.INTER_AREA)
    x_step = w/grid
    y_step = h/grid
    for row in range(grid):
        for column in range(grid):
            img[int(y+row*y_step):int(y+(row+1)*y_step), int(x+column*x_step):int(x+(column+1)*x_step)] = resized[row, column]
    return img


if __name__ == "__main__":
    path = 'TEDx Talk _Will_Stephen.avi'
    capture = cv.VideoCapture(path)
    fps_rate = capture.get(cv.CAP_PROP_FPS)
    file_name, ext = os.path.splitext(path)
    dims = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out = cv.VideoWriter(file_name+'_edited'+ext, cv.VideoWriter_fourcc(*'XVID'), fps_rate, dims)
    haar_cascade_frontal_face = cv.CascadeClassifier('HaarCascadeFrontalFace.xml')
    haar_cascade_profile_face = cv.CascadeClassifier('HaarCascadeProfileFace.xml')

    while True:
        isTrue, frame = capture.read()
        faces_rectangles = list(haar_cascade_frontal_face.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3))
        faces_rectangles.extend(list(haar_cascade_profile_face.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2)))
        for (x, y, w, h) in faces_rectangles:
            frame = censure(frame, x, y, w, h, 10)
        cv.imshow('Preview', frame)
        out.write(frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    out.release()
    cv.destroyAllWindows()
