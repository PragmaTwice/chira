; ModuleID = '../chira/runtime/chirart.cpp'
source_filename = "../chira/runtime/chirart.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"

%"struct.chirart::Var" = type { i64, %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { ptr, ptr }

@stderr = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [22 x i8] c"Assertion failed: %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [17 x i8] c"Unreachable: %s\0A\00", align 1
@.str.2 = private unnamed_addr constant [16 x i8] c"Invalid Var tag\00", align 1
@.str.3 = private unnamed_addr constant [26 x i8] c"Too many closure captures\00", align 1
@.str.4 = private unnamed_addr constant [21 x i8] c"Var is not a closure\00", align 1
@.str.5 = private unnamed_addr constant [21 x i8] c"Var is not a boolean\00", align 1
@.str.6 = private unnamed_addr constant [20 x i8] c"Not implemented yet\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.8 = private unnamed_addr constant [4 x i8] c"%ld\00", align 1

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_ZN7chirart6assertEbPKc(i1 noundef zeroext %0, ptr noundef %1) local_unnamed_addr #0 {
  br i1 %0, label %6, label %3, !prof !5

3:                                                ; preds = %2
  %4 = load ptr, ptr @stderr, align 8, !tbaa !6
  %5 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %4, ptr noundef nonnull @.str, ptr noundef %1) #9
  tail call void @abort() #10
  unreachable

6:                                                ; preds = %2
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: cold mustprogress nofree noreturn nounwind uwtable
define dso_local void @_ZN7chirart11unreachableEPKc(ptr noundef %0) local_unnamed_addr #3 {
  %2 = load ptr, ptr @stderr, align 8, !tbaa !6
  %3 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %2, ptr noundef nonnull @.str.1, ptr noundef %0) #9
  tail call void @abort() #10
  unreachable
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_unspec(ptr nocapture noundef writeonly initializes((0, 8)) %0) local_unnamed_addr #4 {
  store i64 0, ptr %0, align 8, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #5

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_int(ptr nocapture noundef writeonly initializes((0, 16)) %0, i64 noundef %1) local_unnamed_addr #4 {
  store i64 1, ptr %0, align 8, !tbaa !11
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %1, ptr %3, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_closure(ptr nocapture noundef writeonly %0, ptr noundef %1, ptr noundef %2, i64 noundef %3) local_unnamed_addr #0 {
  %5 = icmp ult i64 %3, 65536
  br i1 %5, label %9, label %6, !prof !5

6:                                                ; preds = %4
  %7 = load ptr, ptr @stderr, align 8, !tbaa !6
  %8 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str, ptr noundef nonnull @.str.3) #9
  tail call void @abort() #10
  unreachable

9:                                                ; preds = %4
  %10 = or disjoint i64 %3, 65536
  store i64 %10, ptr %0, align 8, !tbaa !11
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %1, ptr %11, align 8, !tbaa !14
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %2, ptr %12, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_set(ptr nocapture noundef writeonly initializes((0, 8)) %0, ptr nocapture noundef readonly %1) local_unnamed_addr #0 {
  %3 = load i64, ptr %1, align 8, !tbaa !11
  store i64 %3, ptr %0, align 8, !tbaa !11
  switch i64 %3, label %16 [
    i64 1, label %4
    i64 2, label %8
    i64 3, label %12
  ]

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !14
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %6, ptr %7, align 8, !tbaa !14
  br label %31

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load double, ptr %9, align 8, !tbaa !14
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %10, ptr %11, align 8, !tbaa !14
  br label %31

12:                                               ; preds = %2
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i8, ptr %13, align 8, !tbaa !14, !range !15, !noundef !16
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %14, ptr %15, align 8, !tbaa !14
  br label %31

16:                                               ; preds = %2
  %17 = and i64 %3, -65536
  %18 = icmp eq i64 %17, 65536
  br i1 %18, label %19, label %26

19:                                               ; preds = %16
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %21 = load ptr, ptr %20, align 8, !tbaa !14
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %21, ptr %22, align 8, !tbaa !14
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %24 = load ptr, ptr %23, align 8, !tbaa !14
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %24, ptr %25, align 8, !tbaa !14
  br label %31

26:                                               ; preds = %16
  %27 = icmp eq i64 %3, 0
  br i1 %27, label %31, label %28

28:                                               ; preds = %26
  %29 = load ptr, ptr @stderr, align 8, !tbaa !6
  %30 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %29, ptr noundef nonnull @.str, ptr noundef nonnull @.str.2) #9
  tail call void @abort() #10
  unreachable

31:                                               ; preds = %4, %8, %12, %19, %26
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_func_ptr(ptr nocapture noundef readonly %0) local_unnamed_addr #0 {
  %2 = load i64, ptr %0, align 8, !tbaa !11
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %8, label %5, !prof !5

5:                                                ; preds = %1
  %6 = load ptr, ptr @stderr, align 8, !tbaa !6
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str, ptr noundef nonnull @.str.4) #9
  tail call void @abort() #10
  unreachable

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !14
  ret ptr %10
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_caps(ptr nocapture noundef readonly %0) local_unnamed_addr #0 {
  %2 = load i64, ptr %0, align 8, !tbaa !11
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %8, label %5, !prof !5

5:                                                ; preds = %1
  %6 = load ptr, ptr @stderr, align 8, !tbaa !6
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str, ptr noundef nonnull @.str.4) #9
  tail call void @abort() #10
  unreachable

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %10 = load ptr, ptr %9, align 8, !tbaa !14
  ret ptr %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local ptr @chirart_env_load(ptr nocapture noundef readonly %0, i64 noundef %1) local_unnamed_addr #6 {
  %3 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  %4 = load ptr, ptr %3, align 8, !tbaa !17
  ret ptr %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_env_store(ptr nocapture noundef writeonly %0, i64 noundef %1, ptr noundef %2) local_unnamed_addr #4 {
  %4 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  store ptr %2, ptr %4, align 8, !tbaa !17
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local zeroext i1 @chirart_get_bool(ptr nocapture noundef readonly %0) local_unnamed_addr #0 {
  %2 = load i64, ptr %0, align 8, !tbaa !11
  %3 = icmp eq i64 %2, 3
  br i1 %3, label %7, label %4, !prof !5

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !6
  %6 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %5, ptr noundef nonnull @.str, ptr noundef nonnull @.str.5) #9
  tail call void @abort() #10
  unreachable

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load i8, ptr %8, align 8, !tbaa !14, !range !15, !noundef !16
  %10 = trunc nuw i8 %9 to i1
  ret i1 %10
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_add(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #0 {
  %4 = load i64, ptr %1, align 8, !tbaa !11, !noalias !19
  %5 = icmp eq i64 %4, 1
  %6 = load i64, ptr %2, align 8, !noalias !19
  %7 = icmp eq i64 %6, 1
  %8 = select i1 %5, i1 %7, i1 false
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @_ZN7chirart11unreachableEPKc(ptr noundef nonnull @.str.6) #11, !noalias !19
  unreachable

10:                                               ; preds = %3
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load i64, ptr %11, align 8, !tbaa !14, !noalias !19
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %14 = load i64, ptr %13, align 8, !tbaa !14, !noalias !19
  %15 = add nsw i64 %14, %12
  store i64 1, ptr %0, align 8, !tbaa !11
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %15, ptr %16, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_subtract(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #0 {
  %4 = load i64, ptr %1, align 8, !tbaa !11, !noalias !22
  %5 = icmp eq i64 %4, 1
  %6 = load i64, ptr %2, align 8, !noalias !22
  %7 = icmp eq i64 %6, 1
  %8 = select i1 %5, i1 %7, i1 false
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @_ZN7chirart11unreachableEPKc(ptr noundef nonnull @.str.6) #11, !noalias !22
  unreachable

10:                                               ; preds = %3
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load i64, ptr %11, align 8, !tbaa !14, !noalias !22
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %14 = load i64, ptr %13, align 8, !tbaa !14, !noalias !22
  %15 = sub nsw i64 %12, %14
  store i64 1, ptr %0, align 8, !tbaa !11
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %15, ptr %16, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_lt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #0 {
  %4 = load i64, ptr %1, align 8, !tbaa !11, !noalias !25
  %5 = icmp eq i64 %4, 1
  %6 = load i64, ptr %2, align 8, !noalias !25
  %7 = icmp eq i64 %6, 1
  %8 = select i1 %5, i1 %7, i1 false
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @_ZN7chirart11unreachableEPKc(ptr noundef nonnull @.str.6) #11, !noalias !25
  unreachable

10:                                               ; preds = %3
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load i64, ptr %11, align 8, !tbaa !14, !noalias !25
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %14 = load i64, ptr %13, align 8, !tbaa !14, !noalias !25
  %15 = icmp slt i64 %12, %14
  %16 = zext i1 %15 to i8
  store i64 3, ptr %0, align 8, !tbaa !11
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %16, ptr %17, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_display(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #0 {
  %3 = load i64, ptr %1, align 8, !tbaa !11
  %4 = icmp eq i64 %3, 1
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  tail call void @_ZN7chirart11unreachableEPKc(ptr noundef nonnull @.str.6) #11
  unreachable

6:                                                ; preds = %2
  %7 = load ptr, ptr @stdout, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load i64, ptr %8, align 8, !tbaa !14
  %10 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.8, i64 noundef %9) #12
  store i64 0, ptr %0, align 8, !tbaa !11
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #7 {
  %1 = alloca %"struct.chirart::Var", align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1) #12
  store i64 0, ptr %1, align 8, !tbaa !11
  call void @chiracg_main(ptr noundef nonnull %1, ptr noundef null)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1) #12
  ret i32 0
}

declare void @chiracg_main(ptr noundef, ptr noundef) local_unnamed_addr #8

attributes #0 = { mustprogress nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { cold nofree noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { cold mustprogress nofree noreturn nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { cold nounwind }
attributes #10 = { noreturn nounwind }
attributes #11 = { noreturn }
attributes #12 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 20.1.3 (https://github.com/conda-forge/clangdev-feedstock 3e9dfa811865fe27bcd95c0004d27603f2ec4a73)"}
!5 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS8_IO_FILE", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !13, i64 0}
!12 = !{!"_ZTSN7chirart3VarE", !13, i64 0, !9, i64 8}
!13 = !{!"_ZTSN7chirart3Var3TagE", !9, i64 0}
!14 = !{!9, !9, i64 0}
!15 = !{i8 0, i8 2}
!16 = !{}
!17 = !{!18, !18, i64 0}
!18 = !{!"p1 _ZTSN7chirart3VarE", !8, i64 0}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZN7chirartplERKNS_3VarES2_: argument 0"}
!21 = distinct !{!21, !"_ZN7chirartplERKNS_3VarES2_"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"_ZN7chirartmiERKNS_3VarES2_: argument 0"}
!24 = distinct !{!24, !"_ZN7chirartmiERKNS_3VarES2_"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"_ZN7chirartltERKNS_3VarES2_: argument 0"}
!27 = distinct !{!27, !"_ZN7chirartltERKNS_3VarES2_"}
