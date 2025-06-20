; ModuleID = '/home/twice/projects/chira/chira/runtime/chirart.cpp'
source_filename = "/home/twice/projects/chira/chira/runtime/chirart.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"

%"struct.chirart::Var" = type { i64, %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { ptr, ptr }

@.str = private unnamed_addr constant [19 x i8] c"Invalid tag in Var\00", align 1
@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [22 x i8] c"Assertion failed: %s\0A\00", align 1
@.str.2 = private unnamed_addr constant [26 x i8] c"Too many closure captures\00", align 1
@.str.3 = private unnamed_addr constant [21 x i8] c"Var is not a closure\00", align 1
@.str.4 = private unnamed_addr constant [21 x i8] c"Var is not a boolean\00", align 1
@.str.5 = private unnamed_addr constant [20 x i8] c"Not implemented yet\00", align 1
@.str.7 = private unnamed_addr constant [17 x i8] c"Unreachable: %s\0A\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.8 = private unnamed_addr constant [4 x i8] c"%ld\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_unspec(ptr nocapture noundef writeonly initializes((0, 8)) %0) local_unnamed_addr #0 {
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_int(ptr nocapture noundef writeonly initializes((0, 16)) %0, i64 noundef %1) local_unnamed_addr #0 {
  store i64 1, ptr %0, align 8, !tbaa !5
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %1, ptr %3, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_closure(ptr nocapture noundef writeonly %0, ptr noundef %1, ptr noundef %2, i64 noundef %3) local_unnamed_addr #2 {
  %5 = icmp ult i64 %3, 65536
  br i1 %5, label %9, label %6, !prof !11

6:                                                ; preds = %4
  %7 = load ptr, ptr @stderr, align 8, !tbaa !12
  %8 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.2) #8
  tail call void @abort() #9
  unreachable

9:                                                ; preds = %4
  %10 = or disjoint i64 %3, 65536
  store i64 %10, ptr %0, align 8, !tbaa !5
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %1, ptr %11, align 8, !tbaa !10
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %2, ptr %12, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_set(ptr nocapture noundef writeonly initializes((0, 8)) %0, ptr nocapture noundef readonly %1) local_unnamed_addr #2 {
  %3 = load i64, ptr %1, align 8, !tbaa !5
  store i64 %3, ptr %0, align 8, !tbaa !5
  switch i64 %3, label %16 [
    i64 1, label %4
    i64 2, label %8
    i64 3, label %12
  ]

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !10
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %6, ptr %7, align 8, !tbaa !10
  br label %31

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load double, ptr %9, align 8, !tbaa !10
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %10, ptr %11, align 8, !tbaa !10
  br label %31

12:                                               ; preds = %2
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i8, ptr %13, align 8, !tbaa !10, !range !15, !noundef !16
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %14, ptr %15, align 8, !tbaa !10
  br label %31

16:                                               ; preds = %2
  %17 = and i64 %3, -65536
  %18 = icmp eq i64 %17, 65536
  br i1 %18, label %19, label %26

19:                                               ; preds = %16
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %21 = load ptr, ptr %20, align 8, !tbaa !10
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %21, ptr %22, align 8, !tbaa !10
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %24 = load ptr, ptr %23, align 8, !tbaa !10
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %24, ptr %25, align 8, !tbaa !10
  br label %31

26:                                               ; preds = %16
  %27 = icmp eq i64 %3, 0
  br i1 %27, label %31, label %28

28:                                               ; preds = %26
  %29 = load ptr, ptr @stderr, align 8, !tbaa !12
  %30 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %29, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str) #8
  tail call void @abort() #9
  unreachable

31:                                               ; preds = %4, %8, %12, %19, %26
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_func_ptr(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %8, label %5, !prof !11

5:                                                ; preds = %1
  %6 = load ptr, ptr @stderr, align 8, !tbaa !12
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.3) #8
  tail call void @abort() #9
  unreachable

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !10
  ret ptr %10
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_caps(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %8, label %5, !prof !11

5:                                                ; preds = %1
  %6 = load ptr, ptr @stderr, align 8, !tbaa !12
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.3) #8
  tail call void @abort() #9
  unreachable

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %10 = load ptr, ptr %9, align 8, !tbaa !10
  ret ptr %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local ptr @chirart_env_load(ptr nocapture noundef readonly %0, i64 noundef %1) local_unnamed_addr #3 {
  %3 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  %4 = load ptr, ptr %3, align 8, !tbaa !17
  ret ptr %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_env_store(ptr nocapture noundef writeonly %0, i64 noundef %1, ptr noundef %2) local_unnamed_addr #0 {
  %4 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  store ptr %2, ptr %4, align 8, !tbaa !17
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local zeroext i1 @chirart_get_bool(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = icmp eq i64 %2, 3
  br i1 %3, label %7, label %4, !prof !11

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !12
  %6 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %5, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.4) #8
  tail call void @abort() #9
  unreachable

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load i8, ptr %8, align 8, !tbaa !10, !range !15, !noundef !16
  %10 = trunc nuw i8 %9 to i1
  ret i1 %10
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_add(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !19
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !19
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %12, label %9

9:                                                ; preds = %6, %3
  %10 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %11 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %10, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5) #8, !noalias !19
  tail call void @abort() #9, !noalias !19
  unreachable

12:                                               ; preds = %6
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i64, ptr %13, align 8, !tbaa !10, !noalias !19
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %16 = load i64, ptr %15, align 8, !tbaa !10, !noalias !19
  %17 = add nsw i64 %16, %14
  store i64 1, ptr %0, align 8, !tbaa !5
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %17, ptr %18, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_subtract(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !22
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !22
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %12, label %9

9:                                                ; preds = %6, %3
  %10 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !22
  %11 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %10, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5) #8, !noalias !22
  tail call void @abort() #9, !noalias !22
  unreachable

12:                                               ; preds = %6
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i64, ptr %13, align 8, !tbaa !10, !noalias !22
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %16 = load i64, ptr %15, align 8, !tbaa !10, !noalias !22
  %17 = sub nsw i64 %14, %16
  store i64 1, ptr %0, align 8, !tbaa !5
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %17, ptr %18, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_lt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !25
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !25
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %12, label %9

9:                                                ; preds = %6, %3
  %10 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !25
  %11 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %10, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5) #8, !noalias !25
  tail call void @abort() #9, !noalias !25
  unreachable

12:                                               ; preds = %6
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i64, ptr %13, align 8, !tbaa !10, !noalias !25
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %16 = load i64, ptr %15, align 8, !tbaa !10, !noalias !25
  %17 = icmp slt i64 %14, %16
  %18 = zext i1 %17 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %18, ptr %19, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_display(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #2 {
  %3 = load i64, ptr %1, align 8, !tbaa !5
  %4 = icmp eq i64 %3, 1
  br i1 %4, label %8, label %5

5:                                                ; preds = %2
  %6 = load ptr, ptr @stderr, align 8, !tbaa !12
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5) #8
  tail call void @abort() #9
  unreachable

8:                                                ; preds = %2
  %9 = load ptr, ptr @stdout, align 8, !tbaa !12
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10
  %12 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %9, ptr noundef nonnull @.str.8, i64 noundef %11) #10
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = alloca %"struct.chirart::Var", align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1) #10
  store i64 0, ptr %1, align 8, !tbaa !5
  call void @chiracg_main(ptr noundef nonnull %1, ptr noundef null)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1) #10
  ret i32 0
}

declare void @chiracg_main(ptr noundef, ptr noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #6

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #7

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { cold nofree noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { cold nounwind }
attributes #9 = { noreturn nounwind }
attributes #10 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 20.1.3 (https://github.com/conda-forge/clangdev-feedstock 3e9dfa811865fe27bcd95c0004d27603f2ec4a73)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"_ZTSN7chirart3VarE", !7, i64 0, !8, i64 8}
!7 = !{!"_ZTSN7chirart3Var3TagE", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!12 = !{!13, !13, i64 0}
!13 = !{!"p1 _ZTS8_IO_FILE", !14, i64 0}
!14 = !{!"any pointer", !8, i64 0}
!15 = !{i8 0, i8 2}
!16 = !{}
!17 = !{!18, !18, i64 0}
!18 = !{!"p1 _ZTSN7chirart3VarE", !14, i64 0}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZN7chirartplERKNS_3VarES2_: argument 0"}
!21 = distinct !{!21, !"_ZN7chirartplERKNS_3VarES2_"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"_ZN7chirartmiERKNS_3VarES2_: argument 0"}
!24 = distinct !{!24, !"_ZN7chirartmiERKNS_3VarES2_"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"_ZN7chirartltERKNS_3VarES2_: argument 0"}
!27 = distinct !{!27, !"_ZN7chirartltERKNS_3VarES2_"}
