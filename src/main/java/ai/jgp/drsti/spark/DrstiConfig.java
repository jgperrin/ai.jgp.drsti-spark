package ai.jgp.drsti.spark;

public class DrstiConfig {
  private static DrstiConfig instance = null;

  private String path = "/Users/jgp/git/ai.jgp.drsti/public";

  private DrstiConfig() {
  }

  public static String getExportPath() {
    return getInstance().getExportPath0();
  }

  private static DrstiConfig getInstance() {
    if (DrstiConfig.instance == null) {
      DrstiConfig.instance = new DrstiConfig();
    }
    return DrstiConfig.instance;
  }

  private String getExportPath0() {
    return path;
  }
}
