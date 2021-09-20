package ai.jgp.drsti.spark;

public class DrstiConfig {
  private static DrstiConfig instance = null;

  private String path;

  private DrstiConfig() {
    path = System.getenv(DrstiK.ENV_VAR);
    if (path == null) {
      path = "/var/drsti/public";
    }
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

  public static void setExportPath(String path) {
    getInstance().setExportPath0(path);
  }

  private void setExportPath0(String path2) {
    this.path = path;
  }
}
