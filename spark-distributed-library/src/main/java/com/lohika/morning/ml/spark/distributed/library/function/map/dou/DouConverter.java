package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

import java.util.Arrays;
import java.util.Optional;

public class DouConverter {

    private DouConverter() {}

    public static Double transformEnglishLevel(String levelOfEnglish) {
        switch (levelOfEnglish) {
            case "элементарный": {
                return 1D;
            }

            case "ниже среднего": {
                return 2D;
            }

            case "средний": {
                return 3D;
            }

            case "выше среднего": {
                return 4D;
            }

            case "продвинутый": {
                return 5D;
            }

            default: return 0D;
        }
    }

    public static int transformProgrammingLanguage(String language) {
        Optional<ProgrammingLanguage> programmingLanguage =
                Arrays.stream(ProgrammingLanguage.values())
                        .filter(l -> l.getName().equals(language)).findFirst();

        return programmingLanguage.orElse(ProgrammingLanguage.other).getValue();
    }

    public static ProgrammingLanguage transformProgrammingLanguage(Integer index) {
        Optional<ProgrammingLanguage> programmingLanguage =
                Arrays.stream(ProgrammingLanguage.values())
                        .filter(l -> l.getValue().equals(index)).findFirst();

        return programmingLanguage.orElse(ProgrammingLanguage.notDefined);
    }
}
