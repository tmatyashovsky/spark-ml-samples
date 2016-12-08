package com.lohika.morning.ml.spark.driver.service.image;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import org.springframework.stereotype.Component;

@Component
public class ImageService {

    public void processImages(String inputDirectory,
                              String outputDirectoryName,
                              String outputFileName,
                              boolean saveGrayscaleImage) {
        try {
            Path outputDirectoryPath = getOutputDirectoryPath(Paths.get(inputDirectory), outputDirectoryName);

            Files.walk(Paths.get(inputDirectory)).forEach(filePath -> {
                if (Files.isRegularFile(filePath) && filePath.getFileName().toString().endsWith("jpg")) {
                    processImage(filePath, outputDirectoryPath, outputFileName, saveGrayscaleImage);
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private BufferedImage convertToGrayScale(BufferedImage colorImage) {
        BufferedImage image = new BufferedImage(colorImage.getWidth(), colorImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = image.getGraphics();
        graphics.drawImage(colorImage, 0, 0, null);
        graphics.dispose();

        return image;
    }

    private BufferedImage resize(BufferedImage originalImage, int scaledWidth, int scaledHeight) {
        BufferedImage resizedImage = new BufferedImage(scaledWidth, scaledHeight, BufferedImage.TYPE_INT_RGB);
        Graphics graphics = resizedImage.getGraphics();
        graphics.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight, null);
        graphics.dispose();

        return resizedImage;
    }

    private void vectorize(BufferedImage image, Path originalFilePath, Path outputDirectoryPath, String outputFileName) {
        try (PrintWriter out =  new PrintWriter(new BufferedWriter(new FileWriter(outputDirectoryPath.resolve(outputFileName).toFile(), true)))) {
            out.print(originalFilePath.getFileName().toString().contains("cat") ? "1" : "0");

            Raster raster = image.getData();

            for (int x = 0; x < raster.getWidth(); x++) {
                for (int y = 0; y < raster.getHeight(); y++) {
                    out.print(",");
                    out.print(raster.getSample(x, y, 0));
                }
            }

            out.println("");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void processImage(Path originalFilePath, Path outputDirectoryPath, String outputFileName, boolean saveGrayscaleImage) {
        try {
            BufferedImage coloredImage = ImageIO.read(originalFilePath.toFile());
            BufferedImage resizedGrayScaleImage = resize(convertToGrayScale(coloredImage), 100, 100);

            if (saveGrayscaleImage) {
                saveResizedGrayScaleImage(originalFilePath, outputDirectoryPath, resizedGrayScaleImage);
            }

            vectorize(resizedGrayScaleImage, originalFilePath, outputDirectoryPath, outputFileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Path getOutputDirectoryPath(Path filePath, String outputDirectory) throws IOException {
        Path outputDirectoryFullPath = filePath.getParent().resolve(outputDirectory);
        if (!Files.exists(outputDirectoryFullPath)) {
            Files.createDirectory(outputDirectoryFullPath);
        }
        return outputDirectoryFullPath;
    }

    private void saveResizedGrayScaleImage(Path filePath, Path outputDirectoryFullPath, BufferedImage resizedGrayScaleImage) throws IOException {
        String extension = filePath.getFileName().toString().substring(filePath.getFileName().toString().lastIndexOf('.') + 1);

        ImageIO.write(resizedGrayScaleImage, extension, outputDirectoryFullPath.resolve(filePath.getFileName()).toFile());
    }

}
