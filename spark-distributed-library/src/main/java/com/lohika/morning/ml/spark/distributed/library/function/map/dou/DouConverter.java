package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

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

    private static int oneC = 0;
    private static int apl = 1;
    private static int c = 2;
    private static int cSharp = 3;
    private static int cPlusPlus = 4;
    private static int delphi = 5;
    private static int flex = 6;
    private static int haskell = 7;
    private static int java = 8;
    private static int javaScript = 9;
    private static int objectiveC = 10;
    private static int other = 11;
    private static int perl = 12;
    private static int php = 13;
    private static int python = 14;
    private static int ruby = 15;
    private static int scala = 16;
    private static int sql = 17;
    private static int swift = 18;

    public static int transformProgrammingLanguage(String programmingLanguage) {
        if (programmingLanguage != null) {
            switch (programmingLanguage) {
                case "1C": {
                    return oneC;
                }

                case "1С": {
                    return oneC;
                }

                case "APL": {
                    return apl;
                }

                case "C": {
                    return c;
                }

                case "C#/.NET": {
                    return cSharp;
                }

                case "C++": {
                    return cPlusPlus;
                }

                case "Delphi": {
                    return delphi;
                }

                case "Flex/Flash/AIR": {
                    return flex;
                }

                case "Haskell": {
                    return haskell;
                }

                case "Java": {
                    return java;
                }

                case "JavaScript": {
                    return javaScript;
                }

                case "Objective-C": {
                    return objectiveC;
                }

                case "Other": {
                    return other;
                }

                case "Perl": {
                    return perl;
                }

                case "PHP": {
                    return php;
                }

                case "Python": {
                    return python;
                }

                case "Ruby/Rails": {
                    return ruby;
                }

                case "Scala": {
                    return scala;
                }

                case "SQL": {
                    return sql;
                }

                case "Swift": {
                    return swift;
                }

                case "": {
                    return other;
                }

                default: {
                    return -42;
                }
            }
        } else {
            return other;
        }
    }
}
