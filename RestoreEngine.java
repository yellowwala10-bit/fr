package com.facerestore.mobile;

import android.content.Context;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import java.io.*;
import java.util.*;

/**
 * Android port of the deterministic YuNet/grid part of the Python FaceRestore
 * v11.x pipeline.
 *
 * MediaPipe dense-mesh refinement is intentionally optional on Android.
 * When no dense-landmark runtime is bundled, YuNet's 5-point geometry is used
 * with guarded/fallback behavior.
 */
public class RestoreEngine {

    public static class Result {
        public Mat target, baseAligned, baseMask, baseValid,
                aligned, mask, valid, result, userMask, undoMask;

        public double dx, dy, scale = 1, angle, feather = 18;

        public int faceCount = 1;
        public boolean multiFace;
    }

    private static class Face {
        double[] f;
        double score;

        Face(double[] f, double score) {
            this.f = f;
            this.score = score;
        }
    }

    private final String modelPath;

    public RestoreEngine(Context c) throws Exception {
        File f = new File(
                c.getFilesDir(),
                "face_detection_yunet_2026may.onnx"
        );

        if (!f.exists()) {
            try (
                    InputStream in =
                            c.getAssets().open("face_detection_yunet_2026may.onnx");
                    OutputStream out =
                            new FileOutputStream(f)
            ) {
                byte[] b = new byte[65536];

                for (int n; (n = in.read(b)) > 0; ) {
                    out.write(b, 0, n);
                }
            }
        }

        modelPath = f.getAbsolutePath();
    }

    private FaceDetectorYN detector(Size sz) {
        return FaceDetectorYN.create(
                modelPath,
                "",
                sz,
                0.35f,
                0.30f,
                5000
        );
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    /**
     * Convert YuNet output into image coordinates.
     *
     * YuNet coordinates are first divided by the detector scale and then
     * translated by the grid/cell origin.
     */
    private static double[] copyFace(
            float[] row,
            double scale,
            double ox,
            double oy
    ) {
        double[] f = new double[15];

        for (int i = 0; i < 15; i++) {
            f[i] = row[i];
        }

        /*
         * YuNet:
         * 0,1     = x,y
         * 2,3     = width,height
         * 4..13   = five landmark points
         * 14      = confidence
         */
        for (int i = 0; i < 14; i++) {
            f[i] /= scale;
        }

        f[0] += ox;
        f[1] += oy;

        for (int i = 0; i < 5; i++) {
            f[4 + i * 2] += ox;
            f[5 + i * 2] += oy;
        }

        return f;
    }

    private boolean geometryOk(double[] f, Mat im) {
        if (f == null || f.length < 15) {
            return false;
        }

        int w = im.cols();
        int h = im.rows();

        double x = f[0];
        double y = f[1];
        double bw = f[2];
        double bh = f[3];

        if (bw < Math.max(12, w * .025)
                || bh < Math.max(12, h * .025)) {
            return false;
        }

        if (x + bw < 0
                || y + bh < 0
                || x > w
                || y > h) {
            return false;
        }

        double[][] p = lm5(f);

        double eyeY =
                (p[0][1] + p[1][1]) * .5;

        double mouthY =
                (p[3][1] + p[4][1]) * .5;

        if (!(eyeY >= y - .35 * bh
                && eyeY <= y + .75 * bh)) {
            return false;
        }

        if (!(mouthY >= y - .10 * bh
                && mouthY <= y + 1.10 * bh)) {
            return false;
        }

        if (!(eyeY < mouthY)) {
            return false;
        }

        for (double[] q : p) {
            if (q[0] < x - .45 * bw
                    || q[0] > x + 1.45 * bw
                    || q[1] < y - .45 * bh
                    || q[1] > y + 1.45 * bh) {
                return false;
            }
        }

        return true;
    }

    private double candidateScore(
            double[] f,
            Mat im,
            boolean centerPreference
    ) {
        if (!geometryOk(f, im)) {
            return -1e18;
        }

        double area = f[2] * f[3];

        double cx =
                f[0] + f[2] * .5;

        double cy =
                f[1] + f[3] * .5;

        double central =
                1 - Math.min(
                        1,
                        Math.abs(cx - im.cols() * .5)
                                / Math.max(1, im.cols() * .5)
                );

        double conf =
                clamp(f[14], 0, 1);

        double centerTerm =
                centerPreference
                        ? (.35 + .65 * central)
                        : 1;

        double size =
                Math.min(
                        1,
                        area
                                / Math.max(
                                1,
                                im.cols()
                                        * im.rows()
                                        * .08
                        )
                );

        return area
                * centerTerm
                * (.25 + .75 * conf)
                * (.35 + .65 * size);
    }

    private List<Face> detectVariant(
            Mat work,
            double scale,
            double ox,
            double oy,
            boolean centerPreference,
            String mode
    ) {
        List<Face> out =
                new ArrayList<>();

        FaceDetectorYN d =
                detector(work.size());

        Mat faces =
                new Mat();

        d.detect(work, faces);

        for (int r = 0; r < faces.rows(); r++) {

            float[] row =
                    new float[15];

            faces.get(r, 0, row);

            double[] f =
                    copyFace(
                            row,
                            scale,
                            ox,
                            oy
                    );

            out.add(
                    new Face(
                            f,
                            candidateScore(
                                    f,
                                    work,
                                    centerPreference
                            )
                    )
            );
        }

        /*
         * FaceDetectorYN in the OpenCV Java version used by this project
         * does not expose clear().
         *
         * Do NOT call d.clear().
         */
        faces.release();

        return out;
    }

    private List<Face> detectAllPasses(
            Mat im,
            boolean allowBlur
    ) {
        int h = im.rows();
        int w = im.cols();

        double maxSide =
                Math.max(h, w);

        double scale =
                Math.min(
                        1.0,
                        1000.0 / maxSide
                );

        if (maxSide < 700) {
            scale =
                    Math.min(
                            2.0,
                            700.0 / maxSide
                    );
        }

        int ww =
                Math.max(
                        1,
                        (int) Math.round(w * scale)
                );

        int hh =
                Math.max(
                        1,
                        (int) Math.round(h * scale)
                );

        Mat work =
                new Mat();

        Imgproc.resize(
                im,
                work,
                new Size(ww, hh),
                0,
                0,
                scale > 1
                        ? Imgproc.INTER_CUBIC
                        : Imgproc.INTER_AREA
        );

        List<Mat> vars =
                new ArrayList<>();

        vars.add(work);

        Mat gray =
                new Mat();

        Imgproc.cvtColor(
                work,
                gray,
                Imgproc.COLOR_BGR2GRAY
        );

        Mat gray3 =
                new Mat();

        Imgproc.cvtColor(
                gray,
                gray3,
                Imgproc.COLOR_GRAY2BGR
        );

        vars.add(gray3);

        Mat claheGray =
                new Mat();

        Imgproc.createCLAHE(
                2.0,
                new Size(8, 8)
        ).apply(
                gray,
                claheGray
        );

        Mat clahe3 =
                new Mat();

        Imgproc.cvtColor(
                claheGray,
                clahe3,
                Imgproc.COLOR_GRAY2BGR
        );

        vars.add(clahe3);

        if (allowBlur) {

            Mat blur =
                    new Mat();

            Mat sharp =
                    new Mat();

            Imgproc.GaussianBlur(
                    work,
                    blur,
                    new Size(0, 0),
                    1.0
            );

            Core.addWeighted(
                    work,
                    1.35,
                    blur,
                    -.35,
                    0,
                    sharp
            );

            vars.add(sharp);

            blur.release();
        }

        List<Face> out =
                new ArrayList<>();

        for (Mat v : vars) {
            out.addAll(
                    detectVariant(
                            v,
                            scale,
                            0,
                            0,
                            false,
                            "pass"
                    )
            );
        }

        for (Mat v : vars) {
            if (v != work) {
                v.release();
            }
        }

        gray.release();
        claheGray.release();
        work.release();

        return out;
    }

    private List<Face> gridDetect(
            Mat im,
            int cols,
            int rows,
            boolean allowBlur
    ) {
        int h = im.rows();
        int w = im.cols();

        List<Face> out =
                new ArrayList<>();

        double ox =
                .16 / cols;

        double oy =
                .16 / rows;

        for (int r = 0; r < rows; r++) {

            for (int c = 0; c < cols; c++) {

                int x0 =
                        Math.max(
                                0,
                                (int) Math.floor(
                                        (c / (double) cols - ox)
                                                * w
                                )
                        );

                int x1 =
                        Math.min(
                                w,
                                (int) Math.ceil(
                                        ((c + 1) / (double) cols + ox)
                                                * w
                                )
                        );

                int y0 =
                        Math.max(
                                0,
                                (int) Math.floor(
                                        (r / (double) rows - oy)
                                                * h
                                )
                        );

                int y1 =
                        Math.min(
                                h,
                                (int) Math.ceil(
                                        ((r + 1) / (double) rows + oy)
                                                * h
                                )
                        );

                if (x1 <= x0 || y1 <= y0) {
                    continue;
                }

                Mat cell =
                        new Mat(
                                im,
                                new Rect(
                                        x0,
                                        y0,
                                        x1 - x0,
                                        y1 - y0
                                )
                        );

                double scale =
                        Math.min(
                                2.5,
                                Math.max(
                                        1.0,
                                        700.0
                                                / Math.max(
                                                cell.cols(),
                                                cell.rows()
                                        )
                                )
                        );

                Mat work =
                        new Mat();

                Imgproc.resize(
                        cell,
                        work,
                        new Size(
                                Math.max(
                                        1,
                                        (int)
                                                (cell.cols() * scale)
                                ),
                                Math.max(
                                        1,
                                        (int)
                                                (cell.rows() * scale)
                                )
                        ),
                        0,
                        0,
                        scale > 1
                                ? Imgproc.INTER_CUBIC
                                : Imgproc.INTER_AREA
                );

                Mat gray =
                        new Mat();

                Imgproc.cvtColor(
                        work,
                        gray,
                        Imgproc.COLOR_BGR2GRAY
                );

                List<Mat> vars =
                        new ArrayList<>();

                vars.add(work);

                Mat g3 =
                        new Mat();

                Imgproc.cvtColor(
                        gray,
                        g3,
                        Imgproc.COLOR_GRAY2BGR
                );

                vars.add(g3);

                Mat cl =
                        new Mat();

                Imgproc.createCLAHE(
                        2.0,
                        new Size(8, 8)
                ).apply(
                        gray,
                        cl
                );

                Mat cl3 =
                        new Mat();

                Imgproc.cvtColor(
                        cl,
                        cl3,
                        Imgproc.COLOR_GRAY2BGR
                );

                vars.add(cl3);

                if (allowBlur) {

                    Mat bl =
                            new Mat();

                    Mat sh =
                            new Mat();

                    Imgproc.GaussianBlur(
                            work,
                            bl,
                            new Size(0, 0),
                            1.0
                    );

                    Core.addWeighted(
                            work,
                            1.4,
                            bl,
                            -.4,
                            0,
                            sh
                    );

                    vars.add(sh);

                    bl.release();
                }

                Face best =
                        null;

                double ccx =
                        (x0 + x1) * .5;

                double ccy =
                        (y0 + y1) * .5;

                for (Mat v : vars) {

                    FaceDetectorYN d =
                            detector(v.size());

                    Mat fs =
                            new Mat();

                    d.detect(
                            v,
                            fs
                    );

                    for (int i = 0;
                         i < fs.rows();
                         i++) {

                        float[] row =
                                new float[15];

                        fs.get(
                                i,
                                0,
                                row
                        );

                        /*
                         * copyFace divides by detector scale and then
                         * adds x0/y0 exactly once.
                         */
                        double[] f =
                                copyFace(
                                        row,
                                        scale,
                                        x0,
                                        y0
                                );

                        if (!geometryOk(f, im)) {
                            continue;
                        }

                        double fx =
                                f[0] + f[2] * .5;

                        double fy =
                                f[1] + f[3] * .5;

                        double dist =
                                Math.hypot(
                                        (fx - ccx)
                                                / Math.max(
                                                1.0,
                                                w / (double) cols
                                        ),
                                        (fy - ccy)
                                                / Math.max(
                                                1.0,
                                                h / (double) rows
                                        )
                                );

                        double score =
                                clamp(
                                        f[14],
                                        0,
                                        1
                                )
                                        - .18 * dist;

                        if (best == null
                                || score > best.score) {

                            best =
                                    new Face(
                                            f,
                                            score
                                    );
                        }
                    }

                    /*
                     * Do not call d.clear().
                     */
                    fs.release();
                }

                if (best != null) {
                    out.add(best);
                }

                for (Mat v : vars) {
                    if (v != work) {
                        v.release();
                    }
                }

                gray.release();
                cl.release();
                work.release();
                cell.release();
            }
        }

        return dedupe(
                out,
                im,
                cols * rows
        );
    }

    private List<Face> dedupe(
            List<Face> in,
            Mat im,
            int max
    ) {
        List<Face> sorted =
                new ArrayList<>(in);

        sorted.sort(
                (a, b) ->
                        Double.compare(
                                b.score,
                                a.score
                        )
        );

        List<Face> out =
                new ArrayList<>();

        for (Face cand : sorted) {

            double cx =
                    cand.f[0]
                            + cand.f[2] * .5;

            double cy =
                    cand.f[1]
                            + cand.f[3] * .5;

            boolean duplicate =
                    false;

            for (Face q : out) {

                double qx =
                        q.f[0]
                                + q.f[2] * .5;

                double qy =
                        q.f[1]
                                + q.f[3] * .5;

                double d =
                        Math.hypot(
                                cx - qx,
                                cy - qy
                        )
                                / Math.max(
                                10,
                                (
                                        cand.f[2]
                                                + cand.f[3]
                                                + q.f[2]
                                                + q.f[3]
                                ) * .25
                        );

                double areaRatio =
                        (cand.f[2] * cand.f[3])
                                / Math.max(
                                1,
                                q.f[2] * q.f[3]
                        );

                if (d < .45
                        && areaRatio > .35
                        && areaRatio < 2.85) {

                    duplicate = true;
                    break;
                }
            }

            if (!duplicate) {
                out.add(cand);
            }

            if (out.size() >= max) {
                break;
            }
        }

        /*
         * Sort top-to-bottom and then left-to-right.
         * This allows source and target faces to be paired
         * consistently.
         */
        out.sort((a, b) -> {

            double ay =
                    a.f[1] + a.f[3] * .5;

            double by =
                    b.f[1] + b.f[3] * .5;

            double rowTol =
                    Math.max(
                            8,
                            Math.min(
                                    a.f[3],
                                    b.f[3]
                            ) * .65
                    );

            if (Math.abs(ay - by) > rowTol) {
                return Double.compare(
                        ay,
                        by
                );
            }

            return Double.compare(
                    a.f[0] + a.f[2] * .5,
                    b.f[0] + b.f[2] * .5
            );
        });

        return out;
    }

    private List<Face> detectFacesMulti(
            Mat im,
            boolean allowBlur,
            int maxFaces
    ) {
        List<Face> records =
                detectAllPasses(
                        im,
                        allowBlur
                );

        /*
         * Grid fallback is important for 2x3 / 2x4
         * image layouts.
         */
        for (int rows = 2;
             rows <= 4;
             rows++) {

            List<Face> g =
                    gridDetect(
                            im,
                            2,
                            rows,
                            allowBlur
                    );

            if (g.size() > records.size()) {
                records.addAll(g);
            }
        }

        return dedupe(
                records,
                im,
                maxFaces
        );
    }

    private static double[][] lm5(
            double[] f
    ) {
        double[][] p =
                new double[5][2];

        for (int i = 0; i < 5; i++) {

            p[i][0] =
                    f[4 + i * 2];

            p[i][1] =
                    f[5 + i * 2];
        }

        return p;
    }

    private Mat landmarks(
            double[] f
    ) {
        Mat m =
                new Mat(
                        5,
                        2,
                        CvType.CV_32F
                );

        double[][] p =
                lm5(f);

        for (int i = 0; i < 5; i++) {

            m.put(
                    i,
                    0,
                    (float) p[i][0],
                    (float) p[i][1]
            );
        }

        return m;
    }

    private Mat similarity(
            double[] sf,
            double[] tf,
            boolean guarded
    ) {
        if (guarded) {

            double sc =
                    .5 * (
                            tf[2]
                                    / Math.max(
                                    1,
                                    sf[2]
                            )
                                    +
                                    tf[3]
                                            / Math.max(
                                            1,
                                            sf[3]
                                    )
                    );

            double scx =
                    sf[0]
                            + sf[2] * .5;

            double scy =
                    sf[1]
                            + sf[3] * .48;

            double tcx =
                    tf[0]
                            + tf[2] * .5;

            double tcy =
                    tf[1]
                            + tf[3] * .48;

            Mat M =
                    new Mat(
                            2,
                            3,
                            CvType.CV_64F
                    );

            M.put(
                    0,
                    0,
                    sc,
                    0,
                    tcx - sc * scx
            );

            M.put(
                    1,
                    0,
                    0,
                    sc,
                    tcy - sc * scy
            );

            return M;
        }

        Mat srcPts =
                landmarks(sf);

        Mat dstPts =
                landmarks(tf);

        Mat inl =
                new Mat();

        Mat M =
                Calib3d.estimateAffinePartial2D(
                        srcPts,
                        dstPts,
                        inl,
                        Calib3d.LMEDS
                );

        srcPts.release();
        dstPts.release();

        if (M.empty()) {

            double sc =
                    .5 * (
                            tf[2]
                                    / Math.max(
                                    1,
                                    sf[2]
                            )
                                    +
                                    tf[3]
                                            / Math.max(
                                            1,
                                            sf[3]
                                    )
                    );

            double sx =
                    sf[0]
                            + sf[2] / 2;

            double sy =
                    sf[1]
                            + sf[3] / 2;

            double tx =
                    tf[0]
                            + tf[2] / 2;

            double ty =
                    tf[1]
                            + tf[3] / 2;

            M =
                    new Mat(
                            2,
                            3,
                            CvType.CV_64F
                    );

            M.put(
                    0,
                    0,
                    sc,
                    0,
                    tx - sc * sx
            );

            M.put(
                    1,
                    0,
                    0,
                    sc,
                    ty - sc * sy
            );
        }

        inl.release();

        return M;
    }

    private Mat faceMask(
            Size sz,
            double[] f,
            boolean source,
            double scale
    ) {
        int h =
                (int) sz.height;

        int w =
                (int) sz.width;

        double x = f[0];
        double y = f[1];
        double bw = f[2];
        double bh = f[3];

        double cx =
                x + bw * .5;

        double cy =
                y + bh * .50;

        double s =
                Math.max(
                        .75,
                        scale
                );

        Point[] pts =
                new Point[]{

                        new Point(
                                cx,
                                y - .035 * bh
                        ),

                        new Point(
                                x + .14 * bw,
                                y + .045 * bh
                        ),

                        new Point(
                                x + .015 * bw,
                                y + .235 * bh
                        ),

                        new Point(
                                x - .025 * bw,
                                y + .48 * bh
                        ),

                        new Point(
                                x + .045 * bw,
                                y + .72 * bh
                        ),

                        new Point(
                                x + .23 * bw,
                                y + .95 * bh
                        ),

                        new Point(
                                cx,
                                y + 1.14 * bh
                        ),

                        new Point(
                                x + .77 * bw,
                                y + .95 * bh
                        ),

                        new Point(
                                x + .955 * bw,
                                y + .72 * bh
                        ),

                        new Point(
                                x + 1.025 * bw,
                                y + .48 * bh
                        ),

                        new Point(
                                x + .985 * bw,
                                y + .235 * bh
                        ),

                        new Point(
                                x + .86 * bw,
                                y + .045 * bh
                        )
                };

        Mat m =
                Mat.zeros(
                        h,
                        w,
                        CvType.CV_8U
                );

        MatOfPoint poly =
                new MatOfPoint();

        Point[] scaled =
                new Point[pts.length];

        for (int i = 0;
             i < pts.length;
             i++) {

            scaled[i] =
                    new Point(
                            cx
                                    + (
                                    pts[i].x
                                            - cx
                            ) * s,

                            cy
                                    + (
                                    pts[i].y
                                            - cy
                            ) * s
                    );
        }

        poly.fromArray(
                scaled
        );

        Imgproc.fillPoly(
                m,
                Collections.singletonList(poly),
                new Scalar(255)
        );

        int k =
                Math.max(
                        3,
                        (int)
                                (
                                        Math.min(
                                                bw,
                                                bh
                                        ) * .025
                                )
                );

        if (k % 2 == 0) {
            k++;
        }

        Imgproc.GaussianBlur(
                m,
                m,
                new Size(k, k),
                0
        );

        if (source) {

            Mat roi =
                    Mat.zeros(
                            h,
                            w,
                            CvType.CV_8U
                    );

            Imgproc.rectangle(
                    roi,

                    new Point(
                            Math.max(
                                    0,
                                    x - .12 * bw
                            ),

                            Math.max(
                                    0,
                                    y - .10 * bh
                            )
                    ),

                    new Point(
                            Math.min(
                                    w,
                                    x + 1.12 * bw
                            ),

                            Math.min(
                                    h,
                                    y + 1.18 * bh
                            )
                    ),

                    new Scalar(255),
                    -1
            );

            Core.bitwise_and(
                    m,
                    roi,
                    m
            );

            roi.release();
        }

        poly.release();

        return m;
    }

    private Mat expandMask(
            Mat mask,
            double percent
    ) {
        if (percent <= 100) {
            return mask;
        }

        Mat nz =
                new Mat();

        Core.findNonZero(
                mask,
                nz
        );

        if (nz.empty()) {
            nz.release();
            return mask;
        }

        Rect b =
                Imgproc.boundingRect(
                        new MatOfPoint(nz)
                );

        nz.release();

        int radius =
                Math.max(
                        1,
                        (int)
                                (
                                        Math.min(
                                                b.width,
                                                b.height
                                        )
                                                * (
                                                percent / 100.0
                                                        - 1.0
                                        )
                                                * .5
                                )
                );

        radius =
                Math.min(
                        radius,
                        Math.max(
                                1,
                                (int)
                                        (
                                                Math.min(
                                                        b.width,
                                                        b.height
                                                )
                                                        * .16
                                        )
                        )
                );

        int k =
                radius * 2 + 1;

        Mat ker =
                Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        new Size(k, k)
                );

        Mat out =
                new Mat();

        Imgproc.dilate(
                mask,
                out,
                ker
        );

        ker.release();

        return out;
    }

    private double occlusionScore(
            Mat target,
            double[] f
    ) {
        int h =
                target.rows();

        int w =
                target.cols();

        int x0 =
                Math.max(
                        0,
                        (int)
                                (
                                        f[0]
                                                + .12 * f[2]
                                )
                );

        int x1 =
                Math.min(
                        w,
                        (int)
                                (
                                        f[0]
                                                + .88 * f[2]
                                )
                );

        int y0 =
                Math.max(
                        0,
                        (int)
                                (
                                        f[1]
                                                + .10 * f[3]
                                )
                );

        int y1 =
                Math.min(
                        h,
                        (int)
                                (
                                        f[1]
                                                + .90 * f[3]
                                )
                );

        if (x1 <= x0 || y1 <= y0) {
            return 0;
        }

        Mat roi =
                new Mat(
                        target,
                        new Rect(
                                x0,
                                y0,
                                x1 - x0,
                                y1 - y0
                        )
                );

        Mat gray =
                new Mat();

        Imgproc.cvtColor(
                roi,
                gray,
                Imgproc.COLOR_BGR2GRAY
        );

        Mat blur =
                new Mat();

        Imgproc.GaussianBlur(
                gray,
                blur,
                new Size(5, 5),
                0
        );

        Mat edges =
                new Mat();

        Imgproc.Canny(
                blur,
                edges,
                30,
                90
        );

        MatOfDouble mean =
                new MatOfDouble();

        MatOfDouble sd =
                new MatOfDouble();

        Core.meanStdDev(
                gray,
                mean,
                sd
        );

        double std =
                sd.toArray()[0];

        double edgeDensity =
                Core.countNonZero(edges)
                        / (double)
                        Math.max(
                                1,
                                edges.rows()
                                        * edges.cols()
                        );

        double flat =
                clamp(
                        (38 - std) / 38,
                        0,
                        1
                );

        double low =
                clamp(
                        (.045 - edgeDensity)
                                / .045,
                        0,
                        1
                );

        Mat hsv =
                new Mat();

        Imgproc.cvtColor(
                roi,
                hsv,
                Imgproc.COLOR_BGR2HSV
        );

        List<Mat> ch =
                new ArrayList<>();

        Core.split(
                hsv,
                ch
        );

        Mat satMask =
                gt(
                        ch.get(1),
                        150
                );

        double strongSat =
                Core.countNonZero(
                        satMask
                )
                        / (double)
                        Math.max(
                                1,
                                roi.rows()
                                        * roi.cols()
                        );

        double solid =
                clamp(
                        (strongSat - .35)
                                / .50,
                        0,
                        1
                );

        satMask.release();

        for (Mat q : ch) {
            q.release();
        }

        mean.release();
        sd.release();
        roi.release();
        gray.release();
        blur.release();
        edges.release();
        hsv.release();

        return .45 * flat
                + .25 * low
                + .30 * solid;
    }

    private Mat gt(
            Mat src,
            double value
    ) {
        Mat out =
                new Mat();

        Imgproc.threshold(
                src,
                out,
                value,
                255,
                Imgproc.THRESH_BINARY
        );

        return out;
    }

    private double[] inferTargetFromSource(
            double[] sf,
            Mat clear,
            Mat target
    ) {
        double sh =
                clear.rows();

        double sw =
                clear.cols();

        double th =
                target.rows();

        double tw =
                target.cols();

        double[] tf =
                new double[15];

        tf[0] =
                sf[0] / sw * tw;

        tf[1] =
                sf[1] / sh * th;

        tf[2] =
                sf[2] / sw * tw;

        tf[3] =
                sf[3] / sh * th;

        for (int i = 0; i < 5; i++) {

            tf[4 + i * 2] =
                    sf[4 + i * 2]
                            / sw
                            * tw;

            tf[5 + i * 2] =
                    sf[5 + i * 2]
                            / sh
                            * th;
        }

        tf[14] = .55;

        return tf;
    }

    private void addLayer(
            Mat clear,
            Mat target,
            double[] sf,
            double[] tf,
            Mat aligned,
            Mat unionMask,
            Mat unionValid,
            boolean guarded
    ) {
        Mat sm =
                expandMask(
                        faceMask(
                                clear.size(),
                                sf,
                                true,
                                1.18
                        ),
                        115
                );

        Mat tm =
                expandMask(
                        faceMask(
                                target.size(),
                                tf,
                                false,
                                1.18
                        ),
                        115
                );

        double occ =
                occlusionScore(
                        target,
                        tf
                );

        Mat M =
                similarity(
                        sf,
                        tf,
                        guarded || occ >= .55
                );

        Mat layer =
                new Mat();

        Mat valid =
                new Mat();

        Imgproc.warpAffine(
                clear,
                layer,
                M,
                target.size(),
                Imgproc.INTER_LANCZOS4,
                Core.BORDER_CONSTANT,
                new Scalar(0)
        );

        Imgproc.warpAffine(
                sm,
                valid,
                M,
                target.size(),
                Imgproc.INTER_NEAREST,
                Core.BORDER_CONSTANT,
                new Scalar(0)
        );

        Core.min(
                valid,
                tm,
                valid
        );

        /*
         * Keep pixels only where the warped source face mask
         * is valid.
         */
        Mat clean =
                Mat.zeros(
                        target.size(),
                        CvType.CV_8UC3
                );

        layer.copyTo(
                clean,
                valid
        );

        clean.copyTo(
                aligned,
                valid
        );

        Core.max(
                unionValid,
                valid,
                unionValid
        );

        Core.max(
                unionMask,
                valid,
                unionMask
        );

        clean.release();
        layer.release();
        valid.release();
        sm.release();
        tm.release();
        M.release();
    }

    /**
     * Build a single-face or multi-face result.
     */
    public Result build(
            Mat clear,
            Mat target
    ) throws Exception {

        List<Face> sfList =
                detectFacesMulti(
                        clear,
                        false,
                        8
                );

        List<Face> tfList =
                detectFacesMulti(
                        target,
                        true,
                        8
                );

        if (sfList.isEmpty()) {
            throw new Exception(
                    "No reliable face detected in CLEAR image."
            );
        }

        if (tfList.isEmpty()) {
            throw new Exception(
                    "No reliable face detected in TARGET image."
            );
        }

        /*
         * If the target has fewer detections than the clear/source image,
         * infer missing target geometry from source layout.
         */
        if (sfList.size() >= 2
                && tfList.size() < sfList.size()) {

            List<Face> inferred =
                    new ArrayList<>();

            for (Face s : sfList) {

                inferred.add(
                        new Face(
                                inferTargetFromSource(
                                        s.f,
                                        clear,
                                        target
                                ),
                                .55
                        )
                );
            }

            tfList =
                    inferred;
        }

        int n =
                Math.min(
                        Math.min(
                                sfList.size(),
                                tfList.size()
                        ),
                        8
                );

        if (n == 0) {
            throw new Exception(
                    "Could not pair source and target faces."
            );
        }

        Result r =
                new Result();

        r.target =
                target.clone();

        r.userMask =
                Mat.ones(
                        target.size(),
                        CvType.CV_8U
                );

        r.userMask.setTo(
                new Scalar(255)
        );

        r.baseAligned =
                target.clone();

        r.baseValid =
                Mat.zeros(
                        target.size(),
                        CvType.CV_8U
                );

        r.baseMask =
                Mat.zeros(
                        target.size(),
                        CvType.CV_8U
                );

        r.faceCount = n;
        r.multiFace = n > 1;

        for (int i = 0; i < n; i++) {

            double[] sf =
                    sfList.get(i).f;

            double[] tf =
                    tfList.get(i).f;

            double occ =
                    occlusionScore(
                            target,
                            tf
                    );

            addLayer(
                    clear,
                    target,
                    sf,
                    tf,
                    r.baseAligned,
                    r.baseMask,
                    r.baseValid,
                    occ >= .55
                            || tfList.size()
                            < sfList.size()
            );
        }

        if (r.baseMask.empty()
                || Core.countNonZero(
                r.baseMask
        ) == 0) {

            throw new Exception(
                    "No valid face region could be aligned."
            );
        }

        /*
         * Single-face mode keeps Auto Fine.
         *
         * Multi-face mode skips global Auto Fine because every
         * face already has its own transform.
         */
        if (n == 1) {

            autoFine(r);

        } else {

            r.dx = 0;
            r.dy = 0;
            r.scale = 1;
            r.angle = 0;

            apply(r);
            render(r);
        }

        return r;
    }

    private double edgeScore(
            Mat al,
            Mat target,
            Mat mask
    ) {
        Mat pts =
                new Mat();

        Core.findNonZero(
                mask,
                pts
        );

        if (pts.rows() < 40) {
            pts.release();
            return -1e9;
        }

        Rect rr =
                Imgproc.boundingRect(
                        new MatOfPoint(pts)
                );

        pts.release();

        Mat A =
                new Mat(
                        al,
                        rr
                );

        Mat B =
                new Mat(
                        target,
                        rr
                );

        Mat M =
                new Mat(
                        mask,
                        rr
                );

        Mat ag =
                new Mat();

        Mat bg =
                new Mat();

        Imgproc.cvtColor(
                A,
                ag,
                Imgproc.COLOR_BGR2GRAY
        );

        Imgproc.cvtColor(
                B,
                bg,
                Imgproc.COLOR_BGR2GRAY
        );

        Imgproc.GaussianBlur(
                ag,
                ag,
                new Size(7, 7),
                1.4
        );

        Imgproc.GaussianBlur(
                bg,
                bg,
                new Size(7, 7),
                1.4
        );

        Mat ea =
                new Mat();

        Mat eb =
                new Mat();

        Imgproc.Canny(
                ag,
                ea,
                35,
                100
        );

        Imgproc.Canny(
                bg,
                eb,
                35,
                100
        );

        Mat inv =
                new Mat();

        Core.bitwise_not(
                eb,
                inv
        );

        Mat dist =
                new Mat();

        Imgproc.distanceTransform(
                inv,
                dist,
                Imgproc.DIST_L2,
                3
        );

        Mat edgeMask =
                new Mat();

        Core.bitwise_and(
                ea,
                M,
                edgeMask
        );

        Mat nz =
                new Mat();

        Core.findNonZero(
                edgeMask,
                nz
        );

        if (nz.rows() < 15) {

            nz.release();
            edgeMask.release();
            dist.release();
            inv.release();
            ea.release();
            eb.release();
            A.release();
            B.release();
            M.release();
            ag.release();
            bg.release();

            return -1e9;
        }

        double sum = 0;

        for (int i = 0;
             i < nz.rows();
             i++) {

            double[] q =
                    nz.get(
                            i,
                            0
                    );

            sum += Math.min(
                    15,
                    dist.get(
                            (int) q[1],
                            (int) q[0]
                    )[0]
            );
        }

        double result =
                -sum / nz.rows();

        nz.release();
        edgeMask.release();
        dist.release();
        inv.release();
        ea.release();
        eb.release();
        A.release();
        B.release();
        M.release();
        ag.release();
        bg.release();

        return result;
    }

    private double score(
            Result r,
            double dx,
            double dy,
            double sc,
            double ang,
            Mat smallA,
            Mat smallT,
            Mat smallM,
            Mat smallV,
            double ds,
            double cx,
            double cy
    ) {
        Mat T =
                Imgproc.getRotationMatrix2D(
                        new Point(cx, cy),
                        ang,
                        sc
                );

        T.put(
                0,
                2,
                T.get(0, 2)[0]
                        + dx * ds
        );

        T.put(
                1,
                2,
                T.get(1, 2)[0]
                        + dy * ds
        );

        Mat a =
                new Mat();

        Mat m =
                new Mat();

        Mat v =
                new Mat();

        Imgproc.warpAffine(
                smallA,
                a,
                T,
                smallT.size(),
                Imgproc.INTER_LINEAR
        );

        Imgproc.warpAffine(
                smallM,
                m,
                T,
                smallT.size(),
                Imgproc.INTER_NEAREST
        );

        Imgproc.warpAffine(
                smallV,
                v,
                T,
                smallT.size(),
                Imgproc.INTER_NEAREST
        );

        Core.min(
                m,
                v,
                m
        );

        double result =
                edgeScore(
                        a,
                        smallT,
                        m
                );

        T.release();
        a.release();
        m.release();
        v.release();

        return result;
    }

    private void autoFine(
            Result r
    ) {
        Mat pts =
                new Mat();

        Core.findNonZero(
                r.baseMask,
                pts
        );

        if (pts.empty()) {

            pts.release();

            apply(r);
            render(r);

            return;
        }

        Rect br =
                Imgproc.boundingRect(
                        new MatOfPoint(pts)
                );

        pts.release();

        int pad =
                (int)
                        (
                                Math.max(
                                        br.width,
                                        br.height
                                ) * .42
                        );

        int x0 =
                Math.max(
                        0,
                        br.x - pad
                );

        int y0 =
                Math.max(
                        0,
                        br.y - pad
                );

        int x1 =
                Math.min(
                        r.target.cols(),
                        br.x
                                + br.width
                                + pad
                );

        int y1 =
                Math.min(
                        r.target.rows(),
                        br.y
                                + br.height
                                + pad
                );

        Rect roi =
                new Rect(
                        x0,
                        y0,
                        x1 - x0,
                        y1 - y0
                );

        Mat A =
                new Mat(
                        r.baseAligned,
                        roi
                ).clone();

        Mat B =
                new Mat(
                        r.target,
                        roi
                ).clone();

        Mat M =
                new Mat(
                        r.baseMask,
                        roi
                ).clone();

        Mat V =
                new Mat(
                        r.baseValid,
                        roi
                ).clone();

        double ds =
                Math.min(
                        1,
                        320.0
                                / Math.max(
                                B.cols(),
                                B.rows()
                        )
                );

        if (ds < 1) {

            Size z =
                    new Size(
                            Math.max(
                                    1,
                                    (int)
                                            (
                                                    B.cols()
                                                            * ds
                                            )
                            ),
                            Math.max(
                                    1,
                                    (int)
                                            (
                                                    B.rows()
                                                            * ds
                                            )
                            )
                    );

            Imgproc.resize(
                    A,
                    A,
                    z,
                    0,
                    0,
                    Imgproc.INTER_AREA
            );

            Imgproc.resize(
                    B,
                    B,
                    z,
                    0,
                    0,
                    Imgproc.INTER_AREA
            );

            Imgproc.resize(
                    M,
                    M,
                    z,
                    0,
                    0,
                    Imgproc.INTER_NEAREST
            );

            Imgproc.resize(
                    V,
                    V,
                    z,
                    0,
                    0,
                    Imgproc.INTER_NEAREST
            );
        }

        Mat pp =
                new Mat();

        Core.findNonZero(
                M,
                pp
        );

        if (pp.empty()) {

            pp.release();

            apply(r);
            render(r);

            A.release();
            B.release();
            M.release();
            V.release();

            return;
        }

        Rect mb =
                Imgproc.boundingRect(
                        new MatOfPoint(pp)
                );

        pp.release();

        double cx =
                mb.x
                        + mb.width * .5;

        double cy =
                mb.y
                        + mb.height * .5;

        double best =
                -1e9;

        double bdx = 0;
        double bdy = 0;
        double bsc = 1;
        double bang = 0;

        double[] scales =
                {
                        .94,
                        .97,
                        1,
                        1.03,
                        1.06
                };

        double[] angs =
                {
                        -4,
                        -2,
                        0,
                        2,
                        4
                };

        double[] pos =
                {
                        -12,
                        -6,
                        0,
                        6,
                        12
                };

        for (double sc : scales)
            for (double an : angs)
                for (double dy : pos)
                    for (double dx : pos) {

                        double q =
                                score(
                                        r,
                                        dx,
                                        dy,
                                        sc,
                                        an,
                                        A,
                                        B,
                                        M,
                                        V,
                                        ds,
                                        cx,
                                        cy
                                );

                        if (q > best) {

                            best = q;
                            bdx = dx;
                            bdy = dy;
                            bsc = sc;
                            bang = an;
                        }
                    }

        double[] ms =
                {
                        bsc - .02,
                        bsc,
                        bsc + .02
                };

        double[] ma =
                {
                        bang - 1.5,
                        bang,
                        bang + 1.5
                };

        double[] mpx =
                {
                        bdx - 4,
                        bdx,
                        bdx + 4
                };

        double[] mpy =
                {
                        bdy - 4,
                        bdy,
                        bdy + 4
                };

        for (double sc : ms)
            for (double an : ma)
                for (double dy : mpy)
                    for (double dx : mpx) {

                        double q =
                                score(
                                        r,
                                        dx,
                                        dy,
                                        sc,
                                        an,
                                        A,
                                        B,
                                        M,
                                        V,
                                        ds,
                                        cx,
                                        cy
                                );

                        if (q > best) {

                            best = q;
                            bdx = dx;
                            bdy = dy;
                            bsc = sc;
                            bang = an;
                        }
                    }

        double ox = bdx;
        double oy = bdy;
        double os = bsc;
        double oa = bang;

        double[] fs =
                {
                        os - .01,
                        os,
                        os + .01
                };

        double[] fa =
                {
                        oa - .5,
                        oa,
                        oa + .5
                };

        double[] fx =
                {
                        ox - 2,
                        ox,
                        ox + 2
                };

        double[] fy =
                {
                        oy - 2,
                        oy,
                        oy + 2
                };

        for (double sc : fs)
            for (double an : fa)
                for (double dy : fy)
                    for (double dx : fx) {

                        double q =
                                score(
                                        r,
                                        dx,
                                        dy,
                                        sc,
                                        an,
                                        A,
                                        B,
                                        M,
                                        V,
                                        ds,
                                        cx,
                                        cy
                                );

                        if (q > best) {

                            best = q;
                            bdx = dx;
                            bdy = dy;
                            bsc = sc;
                            bang = an;
                        }
                    }

        r.dx = bdx;
        r.dy = bdy;
        r.scale = bsc;
        r.angle = bang;

        apply(r);
        render(r);

        A.release();
        B.release();
        M.release();
        V.release();
    }

    public void apply(
            Result r
    ) {
        Mat pp =
                new Mat();

        Core.findNonZero(
                r.baseMask,
                pp
        );

        if (pp.empty()) {

            pp.release();

            r.aligned =
                    r.baseAligned.clone();

            r.mask =
                    r.baseMask.clone();

            r.valid =
                    r.baseValid.clone();

            return;
        }

        Rect br =
                Imgproc.boundingRect(
                        new MatOfPoint(pp)
                );

        pp.release();

        double cx =
                br.x
                        + br.width * .5;

        double cy =
                br.y
                        + br.height * .5;

        Mat T =
                Imgproc.getRotationMatrix2D(
                        new Point(cx, cy),
                        r.angle,
                        r.scale
                );

        T.put(
                0,
                2,
                T.get(0, 2)[0]
                        + r.dx
        );

        T.put(
                1,
                2,
                T.get(1, 2)[0]
                        + r.dy
        );

        r.aligned =
                new Mat();

        r.mask =
                new Mat();

        r.valid =
                new Mat();

        Imgproc.warpAffine(
                r.baseAligned,
                r.aligned,
                T,
                r.target.size(),
                Imgproc.INTER_LANCZOS4,
                Core.BORDER_CONSTANT,
                new Scalar(0)
        );

        Imgproc.warpAffine(
                r.baseMask,
                r.mask,
                T,
                r.target.size(),
                Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT,
                new Scalar(0)
        );

        Imgproc.warpAffine(
                r.baseValid,
                r.valid,
                T,
                r.target.size(),
                Imgproc.INTER_NEAREST,
                Core.BORDER_CONSTANT,
                new Scalar(0)
        );

        Core.min(
                r.mask,
                r.valid,
                r.mask
        );

        T.release();
    }

    public void render(
            Result r
    ) {
        Mat editedMask =
                new Mat();

        if (r.userMask != null) {

            Core.bitwise_and(
                    r.mask,
                    r.userMask,
                    editedMask
            );

        } else {

            r.mask.copyTo(
                    editedMask
            );
        }

        Mat soft =
                new Mat();

        Imgproc.GaussianBlur(
                editedMask,
                soft,
                new Size(0, 0),
                Math.max(
                        1.0,
                        r.feather
                )
        );

        int rows =
                r.target.rows();

        int cols =
                r.target.cols();

        int pixels =
                rows * cols;

        byte[] dst =
                new byte[pixels * 3];

        byte[] src =
                new byte[pixels * 3];

        byte[] a =
                new byte[pixels];

        byte[] out =
                new byte[pixels * 3];

        r.target.get(
                0,
                0,
                dst
        );

        r.aligned.get(
                0,
                0,
                src
        );

        soft.get(
                0,
                0,
                a
        );

        for (int i = 0;
             i < pixels;
             i++) {

            int alpha =
                    a[i] & 255;

            int inv =
                    255 - alpha;

            int p =
                    i * 3;

            out[p] =
                    (byte)
                            (
                                    (
                                            (src[p] & 255)
                                                    * alpha
                                                    +
                                                    (dst[p] & 255)
                                                            * inv
                                                    + 127
                                    ) / 255
                            );

            out[p + 1] =
                    (byte)
                            (
                                    (
                                            (src[p + 1] & 255)
                                                    * alpha
                                                    +
                                                    (dst[p + 1] & 255)
                                                            * inv
                                                    + 127
                                    ) / 255
                            );

            out[p + 2] =
                    (byte)
                            (
                                    (
                                            (src[p + 2] & 255)
                                                    * alpha
                                                    +
                                                    (dst[p + 2] & 255)
                                                            * inv
                                                    + 127
                                    ) / 255
                            );
        }

        if (r.result != null) {
            r.result.release();
        }

        r.result =
                new Mat(
                        rows,
                        cols,
                        CvType.CV_8UC3
                );

        r.result.put(
                0,
                0,
                out
        );

        editedMask.release();
        soft.release();
    }

    public void saveUndoMask(
            Result r
    ) {
        if (r == null
                || r.userMask == null) {
            return;
        }

        if (r.undoMask != null) {
            r.undoMask.release();
        }

        r.undoMask =
                r.userMask.clone();
    }

    public boolean undoMask(
            Result r
    ) {
        if (r == null
                || r.undoMask == null) {
            return false;
        }

        Mat current =
                r.userMask;

        r.userMask =
                r.undoMask;

        r.undoMask =
                current;

        return true;
    }

    public void paintMask(
            Result r,
            double x,
            double y,
            double radius,
            boolean restore
    ) {
        if (r == null
                || r.userMask == null) {
            return;
        }

        Imgproc.circle(
                r.userMask,
                new Point(x, y),
                (int)
                        Math.max(
                                2,
                                Math.round(radius)
                        ),
                new Scalar(
                        restore
                                ? 255
                                : 0
                ),
                -1,
                Imgproc.LINE_AA,
                0
        );
    }

    public void resetMask(
            Result r
    ) {
        if (r == null) {
            return;
        }

        if (r.userMask != null) {
            r.userMask.release();
        }

        r.userMask =
                Mat.ones(
                        r.target.size(),
                        CvType.CV_8U
                );

        r.userMask.setTo(
                new Scalar(255)
        );
    }
}