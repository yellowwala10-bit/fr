package com.facerestore.mobile;
import android.content.*;import android.graphics.*;import android.util.*;import android.view.*;

public class EditorView extends View {
 public static final int PAN=0,ERASE=1,RESTORE=2;
 Bitmap bmp; Matrix m=new Matrix(); float lastX,lastY; ScaleGestureDetector scale;
 int mode=PAN; float brushRadius=35,curX=-1,curY=-1;
 Paint cursor=new Paint(Paint.ANTI_ALIAS_FLAG);
 public interface BrushListener{void start();void point(float x,float y,boolean restore,float imageRadius);void end();}
 BrushListener brushListener;

 public EditorView(Context c,AttributeSet a){super(c,a);
  cursor.setStyle(Paint.Style.STROKE);cursor.setStrokeWidth(2);cursor.setARGB(230,255,255,255);
  scale=new ScaleGestureDetector(c,new ScaleGestureDetector.SimpleOnScaleGestureListener(){
   public boolean onScale(ScaleGestureDetector d){m.postScale(d.getScaleFactor(),d.getScaleFactor(),d.getFocusX(),d.getFocusY());invalidate();return true;}
  });
 }
 public void setBitmap(Bitmap b){bmp=b;fit();}
 public void setMode(int v){mode=v;curX=curY=-1;invalidate();}
 public int getMode(){return mode;}
 public void setBrushRadius(float v){brushRadius=Math.max(2,v);invalidate();}
 public void setBrushListener(BrushListener v){brushListener=v;}
 public void fit(){if(bmp==null||getWidth()==0)return;m.reset();float s=Math.min((float)getWidth()/bmp.getWidth(),(float)getHeight()/bmp.getHeight());m.postScale(s,s);m.postTranslate((getWidth()-bmp.getWidth()*s)/2f,(getHeight()-bmp.getHeight()*s)/2f);invalidate();}
 protected void onSizeChanged(int w,int h,int ow,int oh){if(bmp!=null)fit();}
 protected void onDraw(Canvas c){super.onDraw(c);if(bmp!=null)c.drawBitmap(bmp,m,null);
  if(mode!=PAN&&curX>=0){float[] v=new float[9];m.getValues(v);float s=(float)Math.sqrt(v[Matrix.MSCALE_X]*v[Matrix.MSCALE_X]+v[Matrix.MSKEW_Y]*v[Matrix.MSKEW_Y]);c.drawCircle(curX,curY,brushRadius*s,cursor);}
 }
 boolean brush(float vx,float vy){
  if(bmp==null||brushListener==null)return false;Matrix inv=new Matrix();if(!m.invert(inv))return false;
  float[] p={vx,vy};inv.mapPoints(p);if(p[0]<0||p[1]<0||p[0]>=bmp.getWidth()||p[1]>=bmp.getHeight())return false;
  float[] v=new float[9];m.getValues(v);
  float viewScale=(float)Math.sqrt(v[Matrix.MSCALE_X]*v[Matrix.MSCALE_X]+v[Matrix.MSKEW_Y]*v[Matrix.MSKEW_Y]);
  float imageRadius=brushRadius/Math.max(0.001f,viewScale);
  brushListener.point(p[0],p[1],mode==RESTORE,imageRadius);return true;
 }
 public boolean onTouchEvent(MotionEvent e){
  final int action=e.getActionMasked();

  // Always allow pinch zoom. After pinch ends, the next finger-down starts
  // a completely fresh brush stroke.
  scale.onTouchEvent(e);
  if(e.getPointerCount()>1 || scale.isInProgress()){
   curX=curY=-1;
   invalidate();
   return true;
  }

  if(mode==PAN){
   if(action==MotionEvent.ACTION_DOWN){
    lastX=e.getX();lastY=e.getY();
   }else if(action==MotionEvent.ACTION_MOVE){
    m.postTranslate(e.getX()-lastX,e.getY()-lastY);
    lastX=e.getX();lastY=e.getY();invalidate();
   }
   return true;
  }

  if(action==MotionEvent.ACTION_DOWN){
   // New independent erase/restore stroke every time the finger touches down.
   curX=e.getX();curY=e.getY();
   if(brushListener!=null)brushListener.start();
   brush(curX,curY);
   invalidate();
   return true;
  }

  if(action==MotionEvent.ACTION_MOVE){
   curX=e.getX();curY=e.getY();
   brush(curX,curY);
   invalidate();
   return true;
  }

  if(action==MotionEvent.ACTION_UP){
   curX=e.getX();curY=e.getY();
   brush(curX,curY);
   if(brushListener!=null)brushListener.end();
   curX=curY=-1;
   invalidate();
   return true;
  }

  if(action==MotionEvent.ACTION_CANCEL){
   if(brushListener!=null)brushListener.end();
   curX=curY=-1;
   invalidate();
   return true;
  }
  return true;
 }
}