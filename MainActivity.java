package com.facerestore.mobile;
import android.app.*;import android.os.*;import android.provider.MediaStore;import android.content.*;import android.graphics.*;import android.net.*;import android.widget.*;import androidx.appcompat.app.AppCompatActivity;
import org.opencv.android.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.*;
public class MainActivity extends AppCompatActivity{
 static{System.loadLibrary("opencv_java4");} final int CLEAR=1,TARGET=2;Mat clear,target;RestoreEngine eng;RestoreEngine.Result rr;EditorView ed;TextView status,xl,yl,sl,rl,fl,bl;SeekBar xs,ys,ss,rs,fs,bs;
 protected void onCreate(Bundle b){super.onCreate(b);setContentView(R.layout.activity_main);ed=findViewById(R.id.editor);status=findViewById(R.id.status);
  try{eng=new RestoreEngine(this);}catch(Exception e){status.setText("MODEL ERROR: "+e.getMessage());}
  findViewById(R.id.clearBtn).setOnClickListener(v->pick(CLEAR));findViewById(R.id.targetBtn).setOnClickListener(v->pick(TARGET));findViewById(R.id.autoBtn).setOnClickListener(v->auto());findViewById(R.id.saveBtn).setOnClickListener(v->save());
  xs=findViewById(R.id.xSeek);ys=findViewById(R.id.ySeek);ss=findViewById(R.id.sizeSeek);rs=findViewById(R.id.rotSeek);fs=findViewById(R.id.featherSeek);bs=findViewById(R.id.brushSeek);
  xl=findViewById(R.id.xLabel);yl=findViewById(R.id.yLabel);sl=findViewById(R.id.sizeLabel);rl=findViewById(R.id.rotLabel);fl=findViewById(R.id.featherLabel);bl=findViewById(R.id.brushLabel);
  SeekBar.OnSeekBarChangeListener l=new SeekBar.OnSeekBarChangeListener(){public void onStartTrackingTouch(SeekBar s){}public void onStopTrackingTouch(SeekBar s){update();}public void onProgressChanged(SeekBar s,int p,boolean u){labels();}};
  xs.setOnSeekBarChangeListener(l);ys.setOnSeekBarChangeListener(l);ss.setOnSeekBarChangeListener(l);rs.setOnSeekBarChangeListener(l);fs.setOnSeekBarChangeListener(l);
   bs.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){public void onStartTrackingTouch(SeekBar s){}public void onStopTrackingTouch(SeekBar s){}public void onProgressChanged(SeekBar s,int p,boolean u){brushLabel();}});
   findViewById(R.id.panBtn).setOnClickListener(v->tool(EditorView.PAN));
   findViewById(R.id.eraseBtn).setOnClickListener(v->tool(EditorView.ERASE));
   findViewById(R.id.restoreBtn).setOnClickListener(v->tool(EditorView.RESTORE));
   findViewById(R.id.resetBtn).setOnClickListener(v->{if(rr!=null){eng.saveUndoMask(rr);eng.resetMask(rr);eng.render(rr);show();status.setText("Mask reset");}});
   findViewById(R.id.undoBtn).setOnClickListener(v->{if(rr!=null&&eng.undoMask(rr)){eng.render(rr);show();status.setText("Undo");}else Toast.makeText(this,"Nothing to undo",Toast.LENGTH_SHORT).show();});
   ed.setBrushListener(new EditorView.BrushListener(){public void start(){if(rr!=null)eng.saveUndoMask(rr);}public void point(float x,float y,boolean restore,float imageRadius){if(rr!=null)eng.paintMask(rr,x,y,imageRadius,restore);}public void end(){if(rr!=null)new Thread(()->{eng.render(rr);runOnUiThread(()->{show();status.setText(ed.getMode()==EditorView.ERASE?"Erase applied":"Restore applied");});}).start();}});
   tool(EditorView.PAN);brushLabel();
 }
 void pick(int r){Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.setType("image/*");i.addCategory(Intent.CATEGORY_OPENABLE);startActivityForResult(i,r);}
 protected void onActivityResult(int r,int c,Intent d){super.onActivityResult(r,c,d);if(c!=RESULT_OK||d==null)return;try{Mat m=read(d.getData());if(r==CLEAR){clear=m;status.setText("Clear loaded");ed.setBitmap(toBmp(clear));}else{target=m;status.setText("Target loaded");ed.setBitmap(toBmp(target));}}catch(Exception e){status.setText(e.toString());}}
 Mat read(Uri u)throws Exception{try(InputStream in=getContentResolver().openInputStream(u)){Bitmap b=BitmapFactory.decodeStream(in);Mat rgba=new Mat();Utils.bitmapToMat(b,rgba);Mat out=new Mat();Imgproc.cvtColor(rgba,out,Imgproc.COLOR_RGBA2BGR);return out;}}
 void auto(){if(clear==null||target==null){Toast.makeText(this,"Load CLEAR and TARGET",Toast.LENGTH_SHORT).show();return;}status.setText("AUTO v11 Multi-Face...");new Thread(()->{try{rr=eng.build(clear,target);runOnUiThread(()->{sync();show();status.setText(String.format("Done  %d face%s  X %.0f  Y %.0f  Size %.0f%%  Rot %.1f°",rr.faceCount,rr.faceCount==1?"":"s",rr.dx,rr.dy,rr.scale*100,rr.angle));});}catch(Exception e){runOnUiThread(()->status.setText("ERROR: "+e.getMessage()));}}).start();}
 void sync(){xs.setProgress((int)Math.round(rr.dx)+120);ys.setProgress((int)Math.round(rr.dy)+120);ss.setProgress((int)Math.round(rr.scale*100)-70);rs.setProgress((int)Math.round(rr.angle*2)+40);fs.setProgress((int)rr.feather-1);labels();}
 void labels(){xl.setText("X "+(xs.getProgress()-120));yl.setText("Y "+(ys.getProgress()-120));sl.setText("Size "+(ss.getProgress()+70)+"%");rl.setText("Rotation "+((rs.getProgress()-40)/2f)+"°");fl.setText("Feather "+(fs.getProgress()+1));}
 float brushSize(){return bs.getProgress()+10f;}
 void brushLabel(){if(bl!=null){bl.setText("Brush "+(int)brushSize());ed.setBrushRadius(brushSize());}}
 void tool(int t){ed.setMode(t);if(t==EditorView.PAN)status.setText("PAN: drag / pinch zoom");else if(t==EditorView.ERASE)status.setText("ERASE: paint unwanted transferred hair/edge");else status.setText("RESTORE: paint face back");}
 void update(){if(rr==null)return;rr.dx=xs.getProgress()-120;rr.dy=ys.getProgress()-120;rr.scale=(ss.getProgress()+70)/100.;rr.angle=(rs.getProgress()-40)/2.;rr.feather=fs.getProgress()+1;new Thread(()->{eng.apply(rr);eng.render(rr);runOnUiThread(this::show);}).start();}
 void show(){ed.setBitmap(toBmp(rr.result));}
 Bitmap toBmp(Mat bgr){Mat rgba=new Mat();Imgproc.cvtColor(bgr,rgba,Imgproc.COLOR_BGR2RGBA);Bitmap b=Bitmap.createBitmap(rgba.cols(),rgba.rows(),Bitmap.Config.ARGB_8888);Utils.matToBitmap(rgba,b);return b;}
 void save(){if(rr==null)return;try{String n="face_restored_"+System.currentTimeMillis()+".png";android.content.ContentValues v=new android.content.ContentValues();v.put(MediaStore.Images.Media.DISPLAY_NAME,n);v.put(MediaStore.Images.Media.MIME_TYPE,"image/png");v.put(MediaStore.Images.Media.RELATIVE_PATH,"Pictures/FaceRestore");Uri u=getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,v);try(OutputStream o=getContentResolver().openOutputStream(u)){toBmp(rr.result).compress(Bitmap.CompressFormat.PNG,100,o);}Toast.makeText(this,"Saved to Pictures/FaceRestore",Toast.LENGTH_LONG).show();}catch(Exception e){status.setText("SAVE ERROR: "+e.getMessage());}}
}