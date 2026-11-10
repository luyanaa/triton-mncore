/* Run a .vsm file through libmnrt (v1 emulator provider).
 *
 * This is the runtime half of the P3 buffer path: the compiler emits a
 * relocatable .vsm, the host binds buffer arguments to DRAM bases, and the
 * program is executed by the project runtime -- mnrt_load_program() ->
 * mnrt_enqueue() -> mnrt_wait() -- instead of driving assemble3/gpfn3
 * directly. MNRT_DUMP names the file the emulator dump is copied to.
 *
 * Usage: mnrt_run FILE.vsm
 */
#include "mnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s FILE.vsm\n", argv[0]);
    return 2;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    perror("open");
    return 2;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = malloc((size_t)n + 1);
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n)
    return 2;
  fclose(f);
  buf[n] = '\0';

  mnrt_info info;
  memset(&info, 0, sizeof info);
  info.struct_size = sizeof info;
  if (mnrt_get_info(&info) != MNRT_OK)
    return 3;

  mnrt_context *ctx = NULL;
  mnrt_program *prog = NULL;
  mnrt_event *ev = NULL;
  if (mnrt_create_context(&info, &ctx) != MNRT_OK)
    return 3;
  if (mnrt_load_program(ctx, buf, (size_t)n, &prog) != MNRT_OK) {
    char e[512];
    mnrt_last_error(e, sizeof e);
    fprintf(stderr, "load_program: %s\n", e);
    return 4;
  }
  if (mnrt_enqueue(ctx, prog, &ev) != MNRT_OK) {
    char e[512];
    mnrt_last_error(e, sizeof e);
    fprintf(stderr, "enqueue: %s\n", e);
    return 5;
  }
  if (mnrt_wait(ev) != MNRT_OK) {
    fprintf(stderr, "wait\n");
    return 6;
  }
  mnrt_release_event(ev);
  mnrt_release_program(prog);
  mnrt_destroy_context(ctx);
  printf("MNRT OK provider=%s\n", info.provider);
  return 0;
}
